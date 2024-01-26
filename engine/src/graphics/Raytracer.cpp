#include <vector>
#include <random>
#include <omp.h>

#include "../../include/graphics/Raytracer.h"

#include "../../include/core/World.h"
#include "../../include/core/Sphere.h"
#include "../../include/core/Ray.h"
#include "../../include/core/glm.h"
#include "../../include/core/BVH.h"

using namespace PhysicsEngine;

static float hit_sphere(const glm::vec3 &center, float radius, const Ray &ray)
{
    glm::vec3 oc = (ray.mOrigin - center);
    float a = glm::dot(ray.mDirection, ray.mDirection);
    float b = 2.0f * glm::dot(oc, ray.mDirection);
    float c = glm::dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0.0f)
    {
        return -1.0f;
    }
    else
    {
        return (-b - glm::sqrt(discriminant)) / (2.0f * a);
    }
}

static float hit_triangle(const Triangle &triangle, const Ray &ray)
{
    constexpr float epsilon = std::numeric_limits<float>::epsilon();

    glm::vec3 edge1 = triangle.mV1 - triangle.mV0;
    glm::vec3 edge2 = triangle.mV2 - triangle.mV0;
    glm::vec3 ray_cross_e2 = glm::cross(ray.mDirection, edge2);
    float det = glm::dot(edge1, ray_cross_e2);

    if (det > -epsilon && det < epsilon)
        return -1.0f; // This ray is parallel to this triangle.

    float inv_det = 1.0f / det;
    glm::vec3 s = ray.mOrigin - triangle.mV0;
    float u = inv_det * glm::dot(s, ray_cross_e2);

    if (u < 0 || u > 1)
        return -1.0f;

    glm::vec3 s_cross_e1 = glm::cross(s, edge1);
    float v = inv_det * glm::dot(ray.mDirection, s_cross_e1);

    if (v < 0 || u + v > 1)
        return -1.0f;

    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = inv_det * glm::dot(edge2, s_cross_e1);

    if (t > epsilon) // ray intersection
    {
        return t;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return -1.0f;
}

// Iterative computeColor for spheres
static glm::vec3 computeColorIterative(const BVH &bvh,
                                       const std::vector<Sphere> &spheres, 
                                       const std::vector<RaytraceMaterial> &materials,
                                       const Ray &ray, 
                                       int maxDepth)
{
    Ray ray2 = ray;

    glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f);

    for (int depth = 0; depth < maxDepth; depth++)
    {
        BVHHit hit = bvh.intersect(ray2);

        int closest_index = -1;
        float closest_t = std::numeric_limits<float>::max();
        for (int i = 0; i < hit.mLeafCount; i++)
        {
            int startIndex = hit.mLeafs[i].mStartIndex;
            int endIndex = hit.mLeafs[i].mStartIndex + hit.mLeafs[i].mIndexCount;
        
            for (int j = startIndex; j < endIndex; j++)
            {
                float t = hit_sphere(spheres[bvh.mPerm[j]].mCentre, spheres[bvh.mPerm[j]].mRadius, ray2);
                if (t > 0.001f && t < closest_t)
                {
                    closest_t = t;
                    closest_index = (int)bvh.mPerm[j];
                }
            }
        }

        if (closest_index >= 0)
        {
            if (materials[closest_index].mType == RaytraceMaterial::MaterialType::DiffuseLight)
            {
                color *= materials[closest_index].mEmissive;
                break;
            }

            glm::vec3 point = ray2.getPoint(closest_t);
            glm::vec3 normal = spheres[closest_index].getNormal(point);

            switch (materials[closest_index].mType)
            {
            case RaytraceMaterial::MaterialType::Lambertian:
                ray2 = RaytraceMaterial::generate_lambertian_ray(point, normal);
                break;
            case RaytraceMaterial::MaterialType::Metallic:
                ray2 = RaytraceMaterial::generate_metallic_ray(point, ray.mDirection, normal,
                                                                               materials[closest_index].mFuzz);
                break;
            case RaytraceMaterial::MaterialType::Dialectric:
                ray2 = RaytraceMaterial::generate_dialectric_ray(point, ray.mDirection, normal,
                                                                             materials[closest_index].mRefractionIndex);
                break;
            }

            color *= materials[closest_index].mAlbedo;
        }
        else
        {
            //color *= glm::vec3(0.0f, 0.0f, 0.0f);
            glm::vec3 unit_direction = glm::normalize(ray2.mDirection);
            float a = 0.5f * (unit_direction.y + 1.0f);
            color *= (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
            break;
        }
    }

    return color;
}

// Iterative computeColor using TLAS and BLAS
static glm::vec3 computeColorIterative(const TLAS &tlas, const std::vector<BLAS> &blas,
                                       const std::vector<RaytraceMaterial> &materials,
                                       const Ray &ray, int maxDepth)
{
    Ray ray2 = ray;

    glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f);

    for (int depth = 0; depth < maxDepth; depth++)
    {
        TLASHit hit = tlas.intersectTLAS(ray2);

        if (hit.blasIndex >= 0)
        {
            /*if (materials[hit.blasIndex].mType == RaytraceMaterial::MaterialType::DiffuseLight)
            {
                color *= materials[hit.blasIndex].mEmissive;
                break;
            }*/

            glm::vec3 normal = glm::normalize(blas[hit.blasIndex].getTriangle(hit.blasHit.mTriIndex).getNormal());
            glm::vec3 point = ray2.getPoint(hit.blasHit.mT);

            switch (materials[hit.blasIndex].mType)
            {
            case RaytraceMaterial::MaterialType::Lambertian:
                ray2 = RaytraceMaterial::generate_lambertian_ray(point, normal);
                break;
            case RaytraceMaterial::MaterialType::Metallic:
                ray2 = RaytraceMaterial::generate_metallic_ray(point, ray.mDirection, normal,
                                                               materials[hit.blasIndex].mFuzz);
                break;
            case RaytraceMaterial::MaterialType::Dialectric:
                ray2 = RaytraceMaterial::generate_dialectric_ray(point, ray.mDirection, normal,
                                                                 materials[hit.blasIndex].mRefractionIndex);
                break;
            }

            color *= materials[hit.blasIndex].mAlbedo;
        }
        else
        {
            // color *= glm::vec3(0.0f, 0.0f, 0.0f);
            glm::vec3 unit_direction = glm::normalize(ray2.mDirection);
            float a = 0.5f * (unit_direction.y + 1.0f);
            color *= (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
            break;
        }
    }

    return color;
}


Raytracer::Raytracer()
{
    //mSamplesPerRay = 0;
}

Raytracer ::~Raytracer()
{

}

void Raytracer::init(World *world)
{
    mWorld = world;
}

void Raytracer::update(Camera *camera)
{
    // Spheres
    /*srand(0);
    int sphereCount = 100;
    std::vector<Sphere> spheres(sphereCount);
    spheres[0] = Sphere(glm::vec3(0.0, -100.5, -1.0f), 100.0f);
    spheres[1] = Sphere(glm::vec3(-1.0, 0.0, -1.0f), 0.5f);
    spheres[2] = Sphere(glm::vec3(-1.0, 0.0, -1.0f), -0.4f);
    spheres[3] = Sphere(glm::vec3(0.0, 0.0, -1.0f), 0.5f);
    spheres[4] = Sphere(glm::vec3(1.0, 0.0, -1.0f), 0.5f);
    spheres[5] = Sphere(glm::vec3(0.75f, 2.25f, -0.5f), 0.7f);
    for (int i = 6; i < sphereCount; i++)
    {
        spheres[i] = Sphere(glm::linearRand(glm::vec3(-20.0f, 0.0f, -20.0f), glm::vec3(20.0f, 0.0f, 20.0f)), glm::linearRand(0.4f, 2.0f));
    }

    std::vector<RaytraceMaterial> materials(sphereCount);
    materials[0].mAlbedo = glm::vec3(0.8f, 0.8f, 0.0f);

    materials[1].mType = RaytraceMaterial::MaterialType::Dialectric;
    materials[1].mAlbedo = glm::vec3(1.0f, 1.0f, 1.0f);
    materials[1].mRefractionIndex = 1.5f;

    materials[2].mType = RaytraceMaterial::MaterialType::Dialectric;
    materials[2].mAlbedo = glm::vec3(1.0f, 1.0f, 1.0f);
    materials[2].mRefractionIndex = 1.5f;

    materials[3].mType = RaytraceMaterial::MaterialType::Lambertian;
    materials[3].mAlbedo = glm::vec3(0.1f, 0.2f, 0.5f);

    materials[4].mType = RaytraceMaterial::MaterialType::Metallic;
    materials[4].mAlbedo = glm::vec3(0.8f, 0.6f, 0.2f);
    materials[4].mFuzz = 0.1f;

    materials[5].mType = RaytraceMaterial::MaterialType::Lambertian;
    materials[5].mAlbedo = glm::vec3(0.1f, 0.2f, 0.5f);

    for (int i = 6; i < sphereCount; i++)
    {
        materials[i].mType =
            (i % 2 == 0) ? RaytraceMaterial::MaterialType::Lambertian : RaytraceMaterial::MaterialType::Metallic;
        materials[i].mAlbedo = glm::linearRand(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
        materials[i].mFuzz = glm::linearRand(0.0f, 0.2f);
    }

    std::vector<AABB> boundingVolumes(sphereCount);
    for (int i = 0; i < sphereCount; i++)
    {
        boundingVolumes[i].mCentre = spheres[i].mCentre;
        boundingVolumes[i].mSize = 2.0f * glm::vec3(spheres[i].mRadius, spheres[i].mRadius, spheres[i].mRadius);
    }*/







    // TLAS and BLAS
    static bool generate_blas = true;
    if (generate_blas)
    {
        Mesh *planeMesh = mWorld->getPrimtiveMesh(PrimitiveType::Plane);
        Mesh *sphereMesh = mWorld->getPrimtiveMesh(PrimitiveType::Sphere);
        Mesh *cubeMesh = mWorld->getPrimtiveMesh(PrimitiveType::Cube);

        std::vector<float> planeVertices = planeMesh->getVertices();
        std::vector<float> sphereVertices = sphereMesh->getVertices();
        std::vector<float> cubeVertices = cubeMesh->getVertices();

        std::vector<unsigned int> planeIndices = planeMesh->getIndices();
        std::vector<unsigned int> sphereIndices = sphereMesh->getIndices();
        std::vector<unsigned int> cubeIndices = cubeMesh->getIndices();

        std::vector<Triangle> planeTriangles(planeIndices.size() / 3);
        std::vector<Triangle> sphereTriangles(sphereIndices.size() / 3);
        std::vector<Triangle> cubeTriangles(cubeIndices.size() / 3);

        for (size_t i = 0; i < planeIndices.size() / 3; i++)
        {
            unsigned int i0 = planeIndices[3 * i + 0];
            unsigned int i1 = planeIndices[3 * i + 1];
            unsigned int i2 = planeIndices[3 * i + 2];

            glm::vec3 v0 = glm::vec3(planeVertices[3 * i0 + 0], planeVertices[3 * i0 + 1], planeVertices[3 * i0 + 2]);
            glm::vec3 v1 = glm::vec3(planeVertices[3 * i1 + 0], planeVertices[3 * i1 + 1], planeVertices[3 * i1 + 2]);
            glm::vec3 v2 = glm::vec3(planeVertices[3 * i2 + 0], planeVertices[3 * i2 + 1], planeVertices[3 * i2 + 2]);

            planeTriangles[i].mV0 = v0;
            planeTriangles[i].mV1 = v1;
            planeTriangles[i].mV2 = v2;
        }

        for (size_t i = 0; i < sphereIndices.size() / 3; i++)
        {
            unsigned int i0 = sphereIndices[3 * i + 0];
            unsigned int i1 = sphereIndices[3 * i + 1];
            unsigned int i2 = sphereIndices[3 * i + 2];

            glm::vec3 v0 = glm::vec3(sphereVertices[3 * i0 + 0], sphereVertices[3 * i0 + 1], sphereVertices[3 * i0 + 2]);
            glm::vec3 v1 = glm::vec3(sphereVertices[3 * i1 + 0], sphereVertices[3 * i1 + 1], sphereVertices[3 * i1 + 2]);
            glm::vec3 v2 = glm::vec3(sphereVertices[3 * i2 + 0], sphereVertices[3 * i2 + 1], sphereVertices[3 * i2 + 2]);

            sphereTriangles[i].mV0 = v0;
            sphereTriangles[i].mV1 = v1;
            sphereTriangles[i].mV2 = v2;
        }

        for (size_t i = 0; i < cubeIndices.size() / 3; i++)
        {
            unsigned int i0 = cubeIndices[3 * i + 0];
            unsigned int i1 = cubeIndices[3 * i + 1];
            unsigned int i2 = cubeIndices[3 * i + 2];

            glm::vec3 v0 =
                glm::vec3(cubeVertices[3 * i0 + 0], cubeVertices[3 * i0 + 1], cubeVertices[3 * i0 + 2]);
            glm::vec3 v1 =
                glm::vec3(cubeVertices[3 * i1 + 0], cubeVertices[3 * i1 + 1], cubeVertices[3 * i1 + 2]);
            glm::vec3 v2 =
                glm::vec3(cubeVertices[3 * i2 + 0], cubeVertices[3 * i2 + 1], cubeVertices[3 * i2 + 2]);

            cubeTriangles[i].mV0 = v0;
            cubeTriangles[i].mV1 = v1;
            cubeTriangles[i].mV2 = v2;
        }

        glm::mat4 planeModel = glm::mat4(1.0f);
        planeModel[0] *= 10.0f;
        planeModel[1] *= 10.0f;
        planeModel[2] *= 10.0f;
        glm::mat4 sphereModelLeft = glm::mat4(1.0f);
        sphereModelLeft[3] = glm::vec4(-2.0f, 1.0f, 0.0f, 1.0f);
        glm::mat4 sphereModelCentre = glm::mat4(1.0f);
        sphereModelCentre[3] = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
        glm::mat4 sphereModelRight = glm::mat4(1.0f);
        sphereModelRight[3] = glm::vec4(2.0f, 1.0f, 0.0f, 1.0f);
        glm::mat4 cubeModel = glm::mat4(1.0f);
        cubeModel[3] = glm::vec4(0.0f, 3.0f, 0.0f, 1.0f);

        mBLAS.resize(5);
        mBLAS[0].allocateBLAS(planeTriangles.size());
        mBLAS[0].buildBLAS(planeTriangles, planeModel, planeTriangles.size());
        mBLAS[1].allocateBLAS(sphereTriangles.size());
        mBLAS[1].buildBLAS(sphereTriangles, sphereModelLeft, sphereTriangles.size());
        mBLAS[2].allocateBLAS(sphereTriangles.size());
        mBLAS[2].buildBLAS(sphereTriangles, sphereModelCentre, sphereTriangles.size());
        mBLAS[3].allocateBLAS(sphereTriangles.size());
        mBLAS[3].buildBLAS(sphereTriangles, sphereModelRight, sphereTriangles.size());
        mBLAS[4].allocateBLAS(cubeTriangles.size());
        mBLAS[4].buildBLAS(cubeTriangles, cubeModel, cubeTriangles.size());

        mMaterials.resize(5);
        mMaterials[0].mType = RaytraceMaterial::MaterialType::Lambertian;
        mMaterials[0].mAlbedo = glm::vec3(0.1f, 0.2f, 0.5f);

        mMaterials[1].mType = RaytraceMaterial::MaterialType::Dialectric;
        mMaterials[1].mAlbedo = glm::vec3(1.0f, 1.0f, 1.0f);
        mMaterials[1].mRefractionIndex = 1.5f;

        mMaterials[2].mType = RaytraceMaterial::MaterialType::Lambertian;
        mMaterials[2].mAlbedo = glm::vec3(0.8f, 0.2f, 0.5f);

        mMaterials[3].mType = RaytraceMaterial::MaterialType::Metallic;
        mMaterials[3].mAlbedo = glm::vec3(0.8f, 0.6f, 0.2f);
        mMaterials[3].mFuzz = 0.0f;

        mMaterials[4].mType = RaytraceMaterial::MaterialType::Metallic;
        mMaterials[4].mAlbedo = glm::vec3(0.75f, 0.75f, 0.75f);
        mMaterials[4].mFuzz = 0.05f;

        generate_blas = false;
    }

    // BVH bvh;
    // bvh.allocateBVH(boundingVolumes.size());
    // bvh.buildBVH(boundingVolumes.data(), boundingVolumes.size());
    mTLAS.allocateTLAS(5);
    mTLAS.buildTLAS(mBLAS.data(), 5);


    /*for (size_t i = 0; i < mBLAS.size(); i++)
    {
        std::cout << "i: " << i << std::endl;
        for (int j = 0; j < (2 * mBLAS[i].mSize - 1); j++)
        {
            if (mBLAS[i].mNodes[j].mIndexCount > 0)
            {
                std::cout << "index count: " << mBLAS[i].mNodes[j].mIndexCount
                          << " start index: " << mBLAS[i].mNodes[j].mLeftOrStartIndex << std::endl;
            }
       
        }
        std::cout << "" << std::endl;
    }*/

    // If camera is moved, re-compute image
    if (/*camera->mSamplesPerRay == 0 ||*/
        camera->moved()) // add a mWorld->getActiveScene()->sceneChangedThisFrame()??
    {
        for (size_t i = 0; i < camera->mImage.size(); i++)
        {
            camera->mImage[i] = 0.0f;
        }
        for (size_t i = 0; i < camera->mSamplesPerRay.size(); i++)
        {
            camera->mSamplesPerRay[i] = 0;
        }
    }

    // Image size
    int width = camera->getNativeGraphicsRaytracingTex()->getWidth();
    int height = camera->getNativeGraphicsRaytracingTex()->getHeight();

    if (width * height * 3 != camera->mImage.size())
    {
        camera->mImage.resize(width * height * 3);
        camera->mSamplesPerRay.resize(width * height);
        for (size_t i = 0; i < camera->mImage.size(); i++)
        {
            camera->mImage[i] = 0.0f;
        }

        for (size_t i = 0; i < camera->mSamplesPerRay.size(); i++)
        {
            camera->mSamplesPerRay[i] = 0;
        }
    }

    // In NDC we use a 2x2x2 box ranging from [-1,1]x[-1,1]x[-1,1]
    float du = 2.0f / 256;
    float dv = 2.0f / 256;

    int max_bounces = 1;
    {
        constexpr int TILE_WIDTH = 8;
        constexpr int TILE_HEIGHT = 8;

        constexpr int TILE_ROWS = 256 / TILE_HEIGHT;
        constexpr int TILE_COLUMNS = 256 / TILE_WIDTH;

        #pragma omp parallel for schedule(dynamic)
        for (int t = 0; t < TILE_ROWS * TILE_COLUMNS; t++)
        {
            int brow = t / TILE_ROWS;
            int bcol = t % TILE_COLUMNS;

            for (int r = 0; r < TILE_HEIGHT; r++)
            {
                for (int c = 0; c < TILE_WIDTH; c++)
                {
                    int row = (TILE_HEIGHT * brow + r);
                    int col = (TILE_WIDTH * bcol + c);

                    glm::vec2 pixelSampleNDC = camera->generatePixelSampleNDC(col, row, du, dv);

                    int irow = (int)(height * (0.5f * (pixelSampleNDC.y + 1.0f)));
                    int icol = (int)(width * (0.5f * (pixelSampleNDC.x + 1.0f)));

                    irow = glm::min(height - 1, glm::max(0, irow));
                    icol = glm::min(width - 1, glm::max(0, icol));

                    int offset = width * irow + icol;

                    assert(offset >= 0);
                    assert(offset < width * height);

                    camera->mSamplesPerRay[offset]++;

                    // Read color from image
                    float red = camera->mImage[3 * offset + 0];
                    float green = camera->mImage[3 * offset + 1];
                    float blue = camera->mImage[3 * offset + 2];
                    glm::vec3 color = glm::vec3(red, green, blue);

                    // color += computeColorIterative(bvh, spheres, materials, camera->getCameraRay(pixelSampleNDC), max_bounces);
                    color += computeColorIterative(mTLAS, mBLAS, mMaterials, camera->getCameraRay(pixelSampleNDC), max_bounces);

                    // Store computed color to image
                    camera->mImage[3 * offset + 0] = color.r;
                    camera->mImage[3 * offset + 1] = color.g;
                    camera->mImage[3 * offset + 2] = color.b;
                }
            }
        }
    }
    
    std::vector<unsigned char> finalImage(3 * width * height);
    #pragma omp parallel for
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            int sampleCount = camera->mSamplesPerRay[width * row + col];
    
            if (sampleCount > 0)
            {
                // Read color from image
                float r = camera->mImage[3 * width * row + 3 * col + 0];
                float g = camera->mImage[3 * width * row + 3 * col + 1];
                float b = camera->mImage[3 * width * row + 3 * col + 2];

                float scale = 1.0f / sampleCount;
                r *= scale;
                g *= scale;
                b *= scale;

                // Gamma correction
                r = glm::sqrt(r);
                g = glm::sqrt(g);
                b = glm::sqrt(b);

                int ir = (int)(255 * glm::clamp(r, 0.0f, 1.0f));
                int ig = (int)(255 * glm::clamp(g, 0.0f, 1.0f));
                int ib = (int)(255 * glm::clamp(b, 0.0f, 1.0f));

                finalImage[3 * width * row + 3 * col + 0] = static_cast<unsigned char>(ir);
                finalImage[3 * width * row + 3 * col + 1] = static_cast<unsigned char>(ig);
                finalImage[3 * width * row + 3 * col + 2] = static_cast<unsigned char>(ib);
            }
            else
            {
                finalImage[3 * width * row + 3 * col + 0] = static_cast<unsigned char>(0);
                finalImage[3 * width * row + 3 * col + 1] = static_cast<unsigned char>(0);
                finalImage[3 * width * row + 3 * col + 2] = static_cast<unsigned char>(0);
            }
        }
    }

    camera->updateRayTracingTexture(finalImage);

    //bvh.freeBVH();
    mTLAS.freeTLAS();
}