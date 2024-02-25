#ifndef RAYTRACER_H__
#define RAYTRACER_H__

#include <vector>

#include "../core/RTGeometry.h"
#include "../core/RaytraceMaterial.h"

namespace PhysicsEngine
{
	class World;
	class Camera;

	class Raytracer
	{
      private:
        World *mWorld;

      public:
        Raytracer();

        void init(World *world);
        void update(Camera *camera, const RTGeometry &geomtry);

	};
}

#endif