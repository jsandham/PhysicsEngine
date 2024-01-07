#ifndef RAYTRACER_H__
#define RAYTRACER_H__

#include <vector>

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