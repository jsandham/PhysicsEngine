#include <vector>
#include <random>

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

// Iterative computeColor for triangles
static glm::vec3 computeColorIterative(const BVH &bvh,
                                       const std::vector<Triangle> &triangles,
                                       const std::vector<RaytraceMaterial> &materials, 
                                       const std::vector<int> &materialPtr, 
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
                float t = hit_triangle(triangles[bvh.mPerm[j]], ray2);
                if (t > 0.001f && t < closest_t)
                {
                    closest_t = t;
                    closest_index = (int)bvh.mPerm[j];
                }
            }
        }

        /*int closest_index = -1;
        float closest_t = std::numeric_limits<float>::max();
        for (size_t i = 0; i < triangles.size(); i++)
        {
            float t = hit_triangle(triangles[i], ray2);
            if (t > 0.0f && t < closest_t)
            {
                closest_t = t;
                closest_index = (int)i;
            }
        }*/

        if (closest_index >= 0)
        {
            int materialIndex = -1;
            for (int i = 0; i < (int)materialPtr.size(); i++)
            {
                if (closest_index >= materialPtr[i] && closest_index < materialPtr[i + 1])
                {
                    materialIndex = i; 
                    break;
                }
            }

            assert(materialIndex >= 0);

            /*if (materials[materialIndex].mType == RaytraceMaterial::MaterialType::DiffuseLight)
            {
                color *= materials[materialIndex].mEmissive;
                break;
            }*/

            glm::vec3 normal = glm::normalize(triangles[closest_index].getNormal());
            glm::vec3 point = ray2.getPoint(closest_t);

            switch (materials[materialIndex].mType)
            {
            case RaytraceMaterial::MaterialType::Lambertian:
                ray2 = RaytraceMaterial::generate_lambertian_ray(point, normal);
                break;
            case RaytraceMaterial::MaterialType::Metallic:
                ray2 = RaytraceMaterial::generate_metallic_ray(point, ray.mDirection, normal,
                                                                             materials[materialIndex].mFuzz);
                break;
            case RaytraceMaterial::MaterialType::Dialectric:
                ray2 = RaytraceMaterial::generate_dialectric_ray(point, ray.mDirection, normal,
                                                                           materials[materialIndex].mRefractionIndex);
                break;
            }

            color *= materials[materialIndex].mAlbedo;
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
    mSamplesPerRay = 0;
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






    //Triangles
    /*srand(0);
    Mesh *sphereMesh = mWorld->getPrimtiveMesh(PrimitiveType::Sphere);
    const std::vector<float> &vertices = sphereMesh->getVertices();
    
    size_t sphereCount = 3;
    size_t triCount = sphereMesh->getVertexCount() / 3;
    std::vector<Triangle> triangles(sphereCount * triCount);
    for (size_t i = 0; i < sphereCount; i++)
    {
        for (size_t j = 0; j < triCount; j++)
        {
            glm::vec3 v0 =
                glm::vec3(vertices[9 * j + 3 * 0 + 0], vertices[9 * j + 3 * 0 + 1], vertices[9 * j + 3 * 0 + 2]);
            glm::vec3 v1 =
                glm::vec3(vertices[9 * j + 3 * 1 + 0], vertices[9 * j + 3 * 1 + 1], vertices[9 * j + 3 * 1 + 2]);
            glm::vec3 v2 =
                glm::vec3(vertices[9 * j + 3 * 2 + 0], vertices[9 * j + 3 * 2 + 1], vertices[9 * j + 3 * 2 + 2]);

            if (i == 1)
            {
                v0 = v0 + glm::vec3(-2.0, 0.0f, 0.0f);
                v1 = v1 + glm::vec3(-2.0, 0.0f, 0.0f);
                v2 = v2 + glm::vec3(-2.0, 0.0f, 0.0f);
            }
            else if (i == 2)
            {
                v0 = v0 + glm::vec3(2.0, 0.0f, 0.0f);
                v1 = v1 + glm::vec3(2.0, 0.0f, 0.0f);
                v2 = v2 + glm::vec3(2.0, 0.0f, 0.0f);
            }

            triangles[triCount * i + j].mV0 = v0;   
            triangles[triCount * i + j].mV1 = v1;
            triangles[triCount * i + j].mV2 = v2;
                
        }
    }

    std::vector<int> materialPtr(sphereCount + 1, 0);
    std::vector<RaytraceMaterial> materials(sphereCount);
    for (size_t i = 0; i < sphereCount; i++)
    {
        RaytraceMaterial::MaterialType type = RaytraceMaterial::MaterialType::Lambertian;
        glm::vec3 albedo = glm::vec3(0.1f, 0.2f, 0.5f);
        if (i == 1)
        {
            albedo = glm::vec3(0.8f, 0.6f, 0.2f);
            type = RaytraceMaterial::MaterialType::Metallic;
        }
        else if (i == 2)
        {
            albedo = glm::vec3(1.0f, 1.0f, 1.0f);
            type = RaytraceMaterial::MaterialType::Dialectric;
        }

        materials[i].mType = type;
        materials[i].mAlbedo = albedo;
        materials[i].mFuzz = 0.1f;
        materials[i].mRefractionIndex = 1.5f;

        materialPtr[i + 1] += materialPtr[i] + (int)triCount;
    }

    std::vector<AABB> boundingVolumes(triangles.size());
    for (int i = 0; i < triangles.size(); i++)
    {
        glm::vec3 bmin = glm::min(triangles[i].mV0, glm::min(triangles[i].mV1, triangles[i].mV2));
        glm::vec3 bmax = glm::max(triangles[i].mV0, glm::max(triangles[i].mV1, triangles[i].mV2));

        glm::vec3 bsize = bmax - bmin;
        bsize.x = glm::max(0.1f, bsize.x);
        bsize.y = glm::max(0.1f, bsize.y);
        bsize.z = glm::max(0.1f, bsize.z);

        boundingVolumes[i].mCentre = bmin + 0.5f * bsize;
        boundingVolumes[i].mSize = bsize;
    }*/







    // TLAS and BLAS
    static bool generate_blas = true;
    if (generate_blas)
    {
        Mesh *planeMesh = mWorld->getPrimtiveMesh(PrimitiveType::Plane);
        Mesh *sphereMesh = mWorld->getPrimtiveMesh(PrimitiveType::Sphere);
    
        std::vector<float> planeVertices = planeMesh->getVertices();
        std::vector<float> sphereVertices = sphereMesh->getVertices();

        std::vector<unsigned int> planeIndices = planeMesh->getIndices();
        std::vector<unsigned int> sphereIndices = sphereMesh->getIndices();

        std::vector<Triangle> planeTriangles(planeIndices.size() / 3);
        std::vector<Triangle> sphereTriangles(sphereIndices.size() / 3);

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

        glm::mat4 planeModel = glm::mat4(1.0f);
        planeModel[0] *= 10.0f;
        planeModel[1] *= 10.0f;
        planeModel[2] *= 10.0f;
        glm::mat4 sphereModelLeft = glm::mat4(1.0f);
        sphereModelLeft[3] = glm::vec4(-2.0f, 2.0f, 0.0f, 1.0f);
        glm::mat4 sphereModelCentre = glm::mat4(1.0f);
        sphereModelCentre[3] = glm::vec4(0.0f, 2.0f, 0.0f, 1.0f);
        glm::mat4 sphereModelRight = glm::mat4(1.0f);
        sphereModelRight[3] = glm::vec4(2.0f, 2.0f, 0.0f, 1.0f);

        mBLAS.resize(4);
        mBLAS[0].allocateBLAS(planeTriangles.size());
        mBLAS[0].buildBLAS(planeTriangles, planeModel, planeTriangles.size());
        mBLAS[1].allocateBLAS(sphereTriangles.size());
        mBLAS[1].buildBLAS(sphereTriangles, sphereModelLeft, sphereTriangles.size());
        mBLAS[2].allocateBLAS(sphereTriangles.size());
        mBLAS[2].buildBLAS(sphereTriangles, sphereModelCentre, sphereTriangles.size());
        mBLAS[3].allocateBLAS(sphereTriangles.size());
        mBLAS[3].buildBLAS(sphereTriangles, sphereModelRight, sphereTriangles.size());

        mMaterials.resize(4);
        mMaterials[0].mType = RaytraceMaterial::MaterialType::Lambertian;
        mMaterials[0].mAlbedo = glm::vec3(0.1f, 0.2f, 0.5f);

        mMaterials[1].mType = RaytraceMaterial::MaterialType::Dialectric;
        mMaterials[1].mAlbedo = glm::vec3(1.0f, 1.0f, 1.0f);
        mMaterials[1].mRefractionIndex = 1.5f;

        mMaterials[2].mType = RaytraceMaterial::MaterialType::Lambertian;
        mMaterials[2].mAlbedo = glm::vec3(0.8f, 0.2f, 0.5f);

        mMaterials[3].mType = RaytraceMaterial::MaterialType::Metallic;
        mMaterials[3].mAlbedo = glm::vec3(0.8f, 0.6f, 0.2f);
        mMaterials[3].mFuzz = 0.1f;

        generate_blas = false;
    }

    mTLAS.allocateTLAS(4);
    mTLAS.buildTLAS(mBLAS.data(), 4);


   

    //BVH bvh;
    //bvh.allocateBVH(boundingVolumes.size());
    //bvh.buildBVH(boundingVolumes.data(), boundingVolumes.size());

    // If camera is moved, re-compute image
    if (mSamplesPerRay == 0 || camera->moved()) // add a mWorld->getActiveScene()->sceneChangedThisFrame()??
    {
        mSamplesPerRay = 0;
        for (size_t i = 0; i < mImage.size(); i++)
        {
            mImage[i] = 0.0f;
        }
    }

    // Image size
    int width = camera->getNativeGraphicsRaytracingTex()->getWidth();
    int height = camera->getNativeGraphicsRaytracingTex()->getHeight();

    if (width * height * 3 != mImage.size())
    {
        mImage.resize(width * height * 3);
        mSamplesPerRay = 0;
        for (size_t i = 0; i < mImage.size(); i++)
        {
            mImage[i] = 0.0f;
        }
    }

    // In NDC we use a 2x2x2 box ranging from [-1,1]x[-1,1]x[-1,1]
    float du = 2.0f / width;
    float dv = 2.0f / height;

    int max_bounces = 50;
    if (mSamplesPerRay < 1000)
    {
        //#pragma omp parallel for schedule(dynamic)
        //for (int row = 0; row < height; row++)
        //{
        //    for (int col = 0; col < width; col++)
        //    {
        //        // Read color from image
        //        float r = mImage[3 * width * row + 3 * col + 0];
        //        float g = mImage[3 * width * row + 3 * col + 1];
        //        float b = mImage[3 * width * row + 3 * col + 2];
        //        glm::vec3 color = glm::vec3(r, g, b);

        //        color += computeColorIterative(spheres, materials, bvh, camera->getCameraRay(col, row, du, dv), max_bounces);
        //        //color +=
        //        //    computeColorIterative(triangles, materials, materialPtr, bvh, camera->getCameraRay(col, row, du, dv), max_bounces);

        //        // Store computed color to image
        //        mImage[3 * width * row + 3 * col + 0] = color.r;
        //        mImage[3 * width * row + 3 * col + 1] = color.g;
        //        mImage[3 * width * row + 3 * col + 2] = color.b;
        //    }
        //}
        int row_dim = 4;
        int col_dim = 4;
        //#pragma omp parallel for schedule(dynamic)
        for (int brow = 0; brow < height / row_dim; brow++)
        {
            for (int bcol = 0; bcol < width / col_dim; bcol++)
            {
                for (int r = 0; r < row_dim; r++)
                {
                    for (int c = 0; c < col_dim; c++)
                    {
                        int row = (row_dim * brow + r);
                        int col = (col_dim * bcol + c);

                        int offset = width * row + col; 

                        // Read color from image
                        float red = mImage[3 * offset + 0];
                        float green = mImage[3 * offset + 1];
                        float blue = mImage[3 * offset + 2];
                        glm::vec3 color = glm::vec3(red, green, blue);

                        //color += computeColorIterative(bvh, spheres, materials, camera->getCameraRay(col, row, du, dv),
                        //                               max_bounces);
                        //color +=
                        //    computeColorIterative(bvh, triangles, materials, materialPtr, camera->getCameraRay(col,
                        //    row, du, dv), max_bounces);
                        color += computeColorIterative(mTLAS, mBLAS, mMaterials, camera->getCameraRay(col, row, du, dv), max_bounces);

                        // Store computed color to image
                        mImage[3 * offset + 0] = color.r;
                        mImage[3 * offset + 1] = color.g;
                        mImage[3 * offset + 2] = color.b;
                    }
                }
            }
        }

        mSamplesPerRay++;
    }
    
    std::vector<unsigned char> finalImage(3 * width * height);
    //#pragma omp parallel for
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            // Read color from image
            float r = mImage[3 * width * row + 3 * col + 0];
            float g = mImage[3 * width * row + 3 * col + 1];
            float b = mImage[3 * width * row + 3 * col + 2];

            float scale = 1.0f / mSamplesPerRay;
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
    }

    camera->updateRayTracingTexture(finalImage);

    //bvh.freeBVH();
    mTLAS.freeTLAS();
}