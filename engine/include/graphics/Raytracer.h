#ifndef RAYTRACER_H__
#define RAYTRACER_H__

#include <vector>

#include "../core/BVH.h"

namespace PhysicsEngine
{
	class World;
	class Camera;
	class Transform;

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
            : mType(MaterialType::Lambertian), mFuzz(0.0f), mRefractionIndex(1.0f),
              mAlbedo(glm::vec3(0.5f, 0.5f, 0.5f)), mEmissive(glm::vec3(0.0f, 0.0f, 0.0f)){};
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

        static Ray generate_dialectric_ray(const glm::vec3 &point, const glm::vec3 &v, const glm::vec3 &normal,
                                           float ir)
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

        static Ray generate_metallic_ray(const glm::vec3 &point, const glm::vec3 &v, const glm::vec3 &normal,
                                         float fuzz)
        {
            Ray ray;
            ray.mOrigin = point;
            ray.mDirection = glm::reflect(glm::normalize(v), normal) +
                             fuzz * glm::vec3(generate_rand(), generate_rand(), generate_rand());

            return ray;
        }

        static Ray generate_lambertian_ray(const glm::vec3 &point, const glm::vec3 &normal)
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

	class Raytracer
	{
      private:
        World *mWorld;

        int mSamplesPerRay;
        std::vector<float> mImage;

        TLAS mTLAS;
        std::vector<BLAS> mBLAS;
        std::vector<RaytraceMaterial> mMaterials;

      public:
        Raytracer();
        ~Raytracer();
        Raytracer(const Raytracer &other) = delete;
        Raytracer &operator=(const Raytracer &other) = delete;

        void init(World *world);
        void update(Camera *camera);

	};
}

#endif