#include <vector>
#include <random>

#include "../../include/graphics/Raytracer.h"

#include "../../include/core/World.h"
#include "../../include/core/Sphere.h"
#include "../../include/core/Ray.h"
#include "../../include/core/glm.h"

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
        return (-b - sqrt(discriminant)) / (2.0f * a);
    }
}


struct RaytraceMaterial
{
    enum class MaterialType
    {
        Lambertian,
        Metallic,
        Dialectric
    };

    MaterialType mType;
    float mFuzz;
    double mRefractionIndex;
    glm::vec3 mAlbedo;

    RaytraceMaterial() : mType(MaterialType::Lambertian), mFuzz(0.0f), mRefractionIndex(1.0f), mAlbedo(glm::vec3(0.5f, 0.5f, 0.5f)){};
    RaytraceMaterial(MaterialType type, float fuzz, double ir, const glm::vec3 &albedo)
        : mType(type), mFuzz(glm::max(0.0f, glm::min(1.0f, fuzz))), mRefractionIndex(ir), mAlbedo(albedo){};

    static float reflectance(float cosine, float ref_idx)
    {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * glm::pow((1 - cosine), 5);
    }

    static glm::vec3 refract(const glm::vec3 &uv, const glm::vec3 &n, float etai_over_etat)
    {
        auto cos_theta = glm::min(glm::dot(-uv, n), 1.0f);
        glm::vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
        glm::vec3 r_out_parallel = -glm::sqrt(glm::abs(1.0f - glm::length2(r_out_perp))) * n;
        return r_out_perp + r_out_parallel;
    }

    static Ray generate_dialectric_ray_on_sphere(const glm::vec3 &point, const glm::vec3 &v, const glm::vec3 &normal, double ir)
    {
        bool front_face = glm::dot(v, normal) < 0.0f;
        float refraction_ratio = front_face ? (1.0f / ir) : ir;

        glm::vec3 normal2 = front_face ? normal : -1.0f * normal;

        glm::vec3 unit_direction = glm::normalize(v);
        float cos_theta = glm::min(glm::dot(-unit_direction, normal2), 1.0f);
        float sin_theta = glm::sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        glm::vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > glm::linearRand(0.0f, 1.0f))
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
        ray.mDirection = glm::reflect(glm::normalize(v), normal) + fuzz * glm::ballRand(1.0f);

        return ray;
    }

    static Ray generate_lambertian_ray_on_hemisphere(const glm::vec3 &point, const glm::vec3 &normal)
    {
        static std::normal_distribution<float> dist(0.0f, 1.0f);
        static std::mt19937 generator;

        glm::vec3 unitSphereVector =
            normal + glm::normalize(glm::vec3(dist(generator), dist(generator), dist(generator)));

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

static glm::vec3 computeColor(const std::vector<Sphere> &spheres, const std::vector<RaytraceMaterial> &materials,
                              const Ray &ray, int depth)
{
    if (depth < 0)
    {
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }

    int closest_index = -1;
    float closest_t = std::numeric_limits<float>::max();
    for (size_t i = 0; i < spheres.size(); i++)
    {
        float t = hit_sphere(spheres[i].mCentre, spheres[i].mRadius, ray);
        if (t > 0.001f && t < closest_t)
        {
            closest_t = t;
            closest_index = (int)i;
        }
    }

    if (closest_index >= 0)
    {
        glm::vec3 point = ray.getPoint(closest_t);
        glm::vec3 normal = glm::normalize((point - spheres[closest_index].mCentre) / spheres[closest_index].mRadius);

        Ray newRay;
        switch (materials[closest_index].mType)
        {
        case RaytraceMaterial::MaterialType::Lambertian:
            newRay = RaytraceMaterial::generate_lambertian_ray_on_hemisphere(point, normal);
            break;
        case RaytraceMaterial::MaterialType::Metallic:
            newRay = RaytraceMaterial::generate_metallic_ray_on_hemisphere(point, ray.mDirection, normal,
                                                                           materials[closest_index].mFuzz);
            break;
        case RaytraceMaterial::MaterialType::Dialectric:
            newRay = RaytraceMaterial::generate_dialectric_ray_on_sphere(point, ray.mDirection, normal, materials[closest_index].mRefractionIndex);
            break;
        }

        return materials[closest_index].mAlbedo * computeColor(spheres, materials, newRay, depth - 1);
    }

    glm::vec3 unit_direction = glm::normalize(ray.mDirection);
    float a = 0.5f * (unit_direction.y + 1.0f);
    return (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
}
/*static glm::vec3 computeColor(const std::vector<Sphere> &spheres, const std::vector<RaytraceMaterial> &materials,
                              const Ray &ray)
{
    Ray ray2 = ray;

    glm::vec3 color = glm::vec3(0.0f, 0.0f, 0.0f);
    float frac = 1.0f;
    
    for (int depth = 0; depth < 10; depth++)
    {
        int closest_index = -1;
        float closest_t = std::numeric_limits<float>::max();
        for (size_t i = 0; i < spheres.size(); i++)
        {
            float t = hit_sphere(spheres[i].mCentre, spheres[i].mRadius, ray2);
            if (t > 0.0f && t < closest_t)
            {
                closest_t = t;
                closest_index = (int)i;
            }
        }

        if (closest_index >= 0)
        {
            glm::vec3 point = ray.getPoint(closest_t);
            glm::vec3 normal = glm::normalize(point - spheres[closest_index].mCentre);

            //Ray newRay;
            if (materials[closest_index].metallic)
            {
                ray2 = generate_metallic_ray_on_hemisphere(point, ray.mDirection, normal);
            }
            else
            {
                ray2 = generate_diffuse_ray_on_hemisphere(point, normal);
            }

            //color += 0.5f * computeColor(spheres, materials, ray2);

            //glm::vec3 unit_direction = glm::normalize(ray2.mDirection);
            //float a = 0.5f * (unit_direction.y + 1.0f);
            //color += (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
            frac *= 0.5f;
        }
        else
        {
            glm::vec3 unit_direction = glm::normalize(ray2.mDirection);
            float a = 0.5f * (unit_direction.y + 1.0f);
            color += (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
            break;
        }
    }

    return frac * color;
}*/

/*static Ray generateRay(const glm::vec3 &eye, const glm::vec3 &pixelCentre, float du, float dv)
{
    static std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    static std::mt19937 generator;

    glm::vec3 pixelSample = pixelCentre + glm::vec3(dist(generator) * du, dist(generator) * dv, 0.0f);

    Ray ray;
    ray.mOrigin = eye;
    ray.mDirection = pixelSample - eye;

    return ray;
}*/

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
    //world.add(make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, material_ground));
    //world.add(make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5, material_center));
    //world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
    //world.add(make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5, material_right));
    std::vector<Sphere> spheres(5);
    spheres[0] = Sphere(glm::vec3(0.0, -100.5, -1.0f), 100.0f);
    spheres[1] = Sphere(glm::vec3(-1.0, 0.0, -1.0f), 0.5f);
    spheres[2] = Sphere(glm::vec3(-1.0, 0.0, -1.0f), -0.4f);
    spheres[3] = Sphere(glm::vec3(0.0, 0.0, -1.0f), 0.5f);
    spheres[4] = Sphere(glm::vec3(1.0, 0.0, -1.0f), 0.5f);


    std::vector<RaytraceMaterial> materials(5);
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

    //materials[4].mType = RaytraceMaterial::MaterialType::Dialectric;
    //materials[4].mAlbedo = glm::vec3(0.6f, 0.5f, 0.7f);
    //materials[4].mRefractionIndex = 1.5f;

    //size_t meshRendererCount = mWorld->getActiveScene()->getNumberOfComponents<MeshRenderer>();
    //
    //std::vector<Sphere> spheres(meshRendererCount);
    //std::vector<RaytraceMaterial> materials(meshRendererCount);
    //for (size_t i = 0; i < meshRendererCount; i++)
    //{
    //    spheres[i].mCentre = mWorld->getActiveScene()->getTransformDataByMeshRendererIndex(i)->mPosition;
    //    spheres[i].mRadius = 1.0f;
    //    materials[i].metallic = (i < 5) ? false : true;
    //}








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

                color += computeColor(spheres, materials, camera->getCameraRay(col, row, du, dv), max_bounces);

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
}