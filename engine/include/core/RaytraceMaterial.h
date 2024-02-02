#ifndef RAYTRACE_MATERIAL_H__
#define RAYTRACE_MATERIAL_H__

#include "glm.h"
#include "Ray.h"

namespace PhysicsEngine
{
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

        RaytraceMaterial()
            : mType(MaterialType::Lambertian), mFuzz(0.0f), mRefractionIndex(1.0f), mAlbedo(glm::vec3(0.5f, 0.5f, 0.5f)),
            mEmissive(glm::vec3(0.0f, 0.0f, 0.0f)) {};
        RaytraceMaterial(MaterialType type, float fuzz, float ir, const glm::vec3& albedo, const glm::vec3& emissive)
            : mType(type), mFuzz(glm::max(0.0f, glm::min(1.0f, fuzz))), mRefractionIndex(ir), mAlbedo(albedo),
            mEmissive(emissive) {};

        static float reflectance(float cosine, float ref_idx)
        {
            // Use Schlick's approximation for reflectance.
            auto r0 = (1.0f - ref_idx) / (1 + ref_idx);
            r0 = r0 * r0;
            return r0 + (1.0f - r0) * glm::pow((1.0f - cosine), 5.0f);
        }

        static glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat)
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

        static float random(float a = 0.0f, float b = 1.0f)
        {
            static uint32_t seed = 1234567;
            seed++;
            float uniform = (float)pcg_hash(seed) / (float)std::numeric_limits<uint32_t>::max();
            return a + (b - a) * uniform;
        }

        static glm::vec3 random_in_unit_sphere()
        {
            while (true)
            {
                glm::vec3 p = glm::vec3(random(-1.0f, 1.0f), random(-1.0f, 1.0f), random(-1.0f, 1.0f));
                if (glm::length2(p) < 1.0f)
                    return p;
            }
        }

        static glm::vec3 random_on_unit_sphere()
        {
            return glm::normalize(random_in_unit_sphere());
        }

        static Ray dialectric_ray(const glm::vec3& point, const glm::vec3& v, const glm::vec3& normal, float ir)
        {
            bool front_face = glm::dot(v, normal) < 0.0f;
            float refraction_ratio = front_face ? (1.0f / ir) : ir;

            glm::vec3 normal2 = front_face ? normal : -1.0f * normal;

            glm::vec3 unit_direction = glm::normalize(v);
            float cos_theta = glm::min(glm::dot(-unit_direction, normal2), 1.0f);
            float sin_theta = glm::sqrt(1.0f - cos_theta * cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
            glm::vec3 direction;

            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random(0.0f, 1.0f))
                direction = glm::reflect(unit_direction, normal2);
            else
                direction = refract(unit_direction, normal2, refraction_ratio);

            Ray ray;
            ray.mOrigin = point;
            ray.mDirection = direction;

            return ray;
        }

        static Ray metallic_ray(const glm::vec3& point, const glm::vec3& v, const glm::vec3& normal, float fuzz)
        {
            glm::vec3 reflected = glm::reflect(glm::normalize(v), normal);
            
            Ray ray;
            ray.mOrigin = point;
            ray.mDirection = reflected + fuzz * random_on_unit_sphere();

            return ray;
        }

        static Ray lambertian_ray(const glm::vec3& point, const glm::vec3& normal)
        {
            glm::vec3 unitSphereVector = glm::normalize(normal + random_on_unit_sphere());

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
}

#endif
