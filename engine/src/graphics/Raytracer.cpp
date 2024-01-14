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


struct RaytraceMaterial
{
    enum class MaterialType
    {
        Lambertian,
        Metallic,
        Dialectric,
        DiffuseLight
    };

    MaterialType mType;
    float mFuzz;
    float mRefractionIndex;
    glm::vec3 mAlbedo;
    glm::vec3 mEmissive;

    RaytraceMaterial() : mType(MaterialType::Lambertian), 
                         mFuzz(0.0f), 
                         mRefractionIndex(1.0f), 
                         mAlbedo(glm::vec3(0.5f, 0.5f, 0.5f)),
                         mEmissive(glm::vec3(0.0f, 0.0f, 0.0f)){};
    RaytraceMaterial(MaterialType type, float fuzz, float ir, const glm::vec3 &albedo, const glm::vec3 &emissive)
        : mType(type), mFuzz(glm::max(0.0f, glm::min(1.0f, fuzz))), mRefractionIndex(ir), mAlbedo(albedo),
          mEmissive(emissive){};

    static float reflectance(float cosine, float ref_idx)
    {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1.0f - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * glm::pow((1.0f - cosine), 5.0f);
    }

    static glm::vec3 refract(const glm::vec3 &uv, const glm::vec3 &n, float etai_over_etat)
    {
        auto cos_theta = glm::min(glm::dot(-uv, n), 1.0f);
        glm::vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
        glm::vec3 r_out_parallel = -glm::sqrt(glm::abs(1.0f - glm::length2(r_out_perp))) * n;
        return r_out_perp + r_out_parallel;
    }

    static uint32_t pcg_hash(uint32_t seed)
    {
        uint32_t state = seed * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    static float generate_rand(float a = 0.0f, float b = 1.0f)
    {
        static uint32_t seed = 1234567;
        seed++;
        float uniform = (float)pcg_hash(seed) / (float)std::numeric_limits<uint32_t>::max();
        return a + (b - a) * uniform;
    }

    static Ray generate_dialectric_ray_on_sphere(const glm::vec3 &point, const glm::vec3 &v, const glm::vec3 &normal, float ir)
    {
        bool front_face = glm::dot(v, normal) < 0.0f;
        float refraction_ratio = front_face ? (1.0f / ir) : ir;

        glm::vec3 normal2 = front_face ? normal : -1.0f * normal;

        glm::vec3 unit_direction = glm::normalize(v);
        float cos_theta = glm::min(glm::dot(-unit_direction, normal2), 1.0f);
        float sin_theta = glm::sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        glm::vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > generate_rand(0.0f, 1.0f))
            direction = glm::reflect(unit_direction, normal2);
        else
            direction = refract(unit_direction, normal2, refraction_ratio);

        Ray ray;
        ray.mOrigin = point;
        ray.mDirection = direction;

        return ray;
    }

    static Ray generate_metallic_ray_on_hemisphere(const glm::vec3 &point, const glm::vec3 &v, const glm::vec3 &normal, float fuzz)
    {
        Ray ray;
        ray.mOrigin = point;
        ray.mDirection = glm::reflect(glm::normalize(v), normal) + fuzz * glm::vec3(generate_rand(), generate_rand(), generate_rand());

        return ray;
    }

    static Ray generate_lambertian_ray_on_hemisphere(const glm::vec3 &point, const glm::vec3 &normal)
    {
        glm::vec3 unitSphereVector =
            normal + glm::normalize(glm::vec3(generate_rand(), generate_rand(), generate_rand()));

        Ray ray;
        ray.mOrigin = point;

        if (glm::dot(unitSphereVector, normal) > 0.0f)
        {
            ray.mDirection = unitSphereVector;
        }
        else
        {
            ray.mDirection = -unitSphereVector;
        }

        return ray;
    }
};

// Iterative computeColor
static glm::vec3 computeColorIterative(const std::vector<Sphere> &spheres, const std::vector<RaytraceMaterial> &materials,
                              const BVH &bvh, const Ray &ray, int maxDepth)
{
    Ray ray2 = ray;

    glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f);

    //std::queue<int> queue;
    int top = 0;
    int stack[32];

    for (int depth = 0; depth < maxDepth; depth++)
    {
        int closest_index = -1;
        float closest_t = std::numeric_limits<float>::max();

        //assert(queue.empty());
        //queue.push(0);
        assert(top == 0);
        stack[0] = 0;
        top++;

        //while (!queue.empty())
        while (top > 0)
        {
            //int nodeIndex = queue.front();
            //queue.pop();
            int nodeIndex = stack[top - 1];
            top--;

            const BVHNode *node = &bvh.mNodes[nodeIndex];

            if (Intersect::intersect(ray2, node->mMin, node->mMax))
            {
                if (!node->isLeaf())
                {
                    //queue.push(node->mLeftOrStartIndex);
                    //queue.push(node->mLeftOrStartIndex + 1);
                    stack[top++] = node->mLeftOrStartIndex;
                    stack[top++] = node->mLeftOrStartIndex + 1;
                }
                else
                {
                    int startIndex = node->mLeftOrStartIndex;
                    int endIndex = node->mLeftOrStartIndex + node->mIndexCount;
                    for (int i = startIndex; i < endIndex; i++)
                    {
                        float t = hit_sphere(spheres[bvh.mPerm[i]].mCentre, spheres[bvh.mPerm[i]].mRadius, ray2);
                        if (t > 0.001f && t < closest_t)
                        {
                            closest_t = t;
                            closest_index = (int)bvh.mPerm[i];
                        }
                    }
                }
            }
        }

        /*int closest_index = -1;
        float closest_t = std::numeric_limits<float>::max();
        for (size_t i = 0; i < spheres.size(); i++)
        {
            float t = hit_sphere(spheres[i].mCentre, spheres[i].mRadius, ray2);
            if (t > 0.0f && t < closest_t)
            {
                closest_t = t;
                closest_index = (int)i;
            }
        }*/

        if (closest_index >= 0)
        {
            if (materials[closest_index].mType == RaytraceMaterial::MaterialType::DiffuseLight)
            {
                color *= materials[closest_index].mEmissive;
                break;
            }

            glm::vec3 point = ray2.getPoint(closest_t);
            glm::vec3 normal =
                glm::normalize((point - spheres[closest_index].mCentre) / spheres[closest_index].mRadius);

            switch (materials[closest_index].mType)
            {
            case RaytraceMaterial::MaterialType::Lambertian:
                ray2 = RaytraceMaterial::generate_lambertian_ray_on_hemisphere(point, normal);
                break;
            case RaytraceMaterial::MaterialType::Metallic:
                ray2 = RaytraceMaterial::generate_metallic_ray_on_hemisphere(point, ray.mDirection, normal,
                                                                               materials[closest_index].mFuzz);
                break;
            case RaytraceMaterial::MaterialType::Dialectric:
                ray2 = RaytraceMaterial::generate_dialectric_ray_on_sphere(point, ray.mDirection, normal,
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



static glm::vec3 computeColorIterative(const std::vector<Triangle> &triangles,
                                       const std::vector<RaytraceMaterial> &materials, const std::vector<int> &materialPtr, const BVH &bvh, const Ray &ray,
                                       int maxDepth)
{
    Ray ray2 = ray;

    glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f);

    // std::queue<int> queue;
    int top = 0;
    int stack[32];

    for (int depth = 0; depth < maxDepth; depth++)
    {
        int closest_index = -1;
        float closest_t = std::numeric_limits<float>::max();

        // assert(queue.empty());
        // queue.push(0);
        assert(top == 0);
        stack[0] = 0;
        top++;

        // while (!queue.empty())
        while (top > 0)
        {
            // mm = glm::max(mm, (int)queue.size());
            // int nodeIndex = queue.front();
            // queue.pop();
            int nodeIndex = stack[top - 1];
            top--;

            const BVHNode *node = &bvh.mNodes[nodeIndex];

            if (Intersect::intersect(ray2, node->mMin, node->mMax))
            {
                if (!node->isLeaf())
                {
                    // queue.push(node->mLeftOrStartIndex);
                    // queue.push(node->mLeftOrStartIndex + 1);
                    stack[top++] = node->mLeftOrStartIndex;
                    stack[top++] = node->mLeftOrStartIndex + 1;
                }
                else
                {
                    int startIndex = node->mLeftOrStartIndex;
                    int endIndex = node->mLeftOrStartIndex + node->mIndexCount;
                    for (int i = startIndex; i < endIndex; i++)
                    {
                        float t = hit_triangle(triangles[bvh.mPerm[i]], ray2);
                        if (t > 0.001f && t < closest_t)
                        {
                            closest_t = t;
                            closest_index = (int)bvh.mPerm[i];
                        }
                    }
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
                ray2 = RaytraceMaterial::generate_lambertian_ray_on_hemisphere(point, normal);
                break;
            case RaytraceMaterial::MaterialType::Metallic:
                ray2 = RaytraceMaterial::generate_metallic_ray_on_hemisphere(point, ray.mDirection, normal,
                                                                             materials[materialIndex].mFuzz);
                break;
            case RaytraceMaterial::MaterialType::Dialectric:
                ray2 = RaytraceMaterial::generate_dialectric_ray_on_sphere(point, ray.mDirection, normal,
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
    srand(0);
    int sphereCount = 6;
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
    }





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










   

    BVH bvh;
    bvh.allocateBVH(boundingVolumes.size());
    bvh.buildBVH(boundingVolumes.data(), boundingVolumes.size());

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
        #pragma omp parallel for
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                // Read color from image
                float r = mImage[3 * width * row + 3 * col + 0];
                float g = mImage[3 * width * row + 3 * col + 1];
                float b = mImage[3 * width * row + 3 * col + 2];
                glm::vec3 color = glm::vec3(r, g, b);

                color += computeColorIterative(spheres, materials, bvh, camera->getCameraRay(col, row, du, dv), max_bounces);
                //color +=
                //    computeColorIterative(triangles, materials, materialPtr, bvh, camera->getCameraRay(col, row, du, dv), max_bounces);

                // Store computed color to image
                mImage[3 * width * row + 3 * col + 0] = color.r;
                mImage[3 * width * row + 3 * col + 1] = color.g;
                mImage[3 * width * row + 3 * col + 2] = color.b;
            }
        }

        mSamplesPerRay++;
    }
    
    std::vector<unsigned char> finalImage(3 * width * height);
    #pragma omp parallel for
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

    bvh.freeBVH();
}