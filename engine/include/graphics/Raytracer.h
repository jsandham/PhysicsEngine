#ifndef RAYTRACER_H__
#define RAYTRACER_H__

#include <vector>

#include "../core/BVH.h"

namespace PhysicsEngine
{
	class World;
	class Camera;
	class Transform;

	class Raytracer
	{
      private:
        World *mWorld;

        int mSamplesPerRay;
        std::vector<float> mImage;

        std::vector<BVH> mBVHs;

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