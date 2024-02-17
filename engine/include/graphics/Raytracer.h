#ifndef RAYTRACER_H__
#define RAYTRACER_H__

#include <vector>

#include "../core/BVH.h"
#include "../core/RaytraceMaterial.h"

namespace PhysicsEngine
{
	class World;
	class Camera;;

	class Raytracer
	{
      private:
        World *mWorld;

      public:
        Raytracer();
        ~Raytracer();
        Raytracer(const Raytracer &other) = delete;
        Raytracer &operator=(const Raytracer &other) = delete;

        void init(World *world);
        void update(Camera *camera, const TLAS &tlas, const std::vector<BLAS*> &blas, const std::vector<glm::mat4> &models, const BVH & bvh, const std::vector<Sphere> &spheres);

	};
}

#endif