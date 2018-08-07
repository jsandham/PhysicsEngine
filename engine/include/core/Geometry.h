#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include "Ray.h"
#include "Bounds.h"
#include "Sphere.h"
#include "Capsule.h"

namespace PhysicsEngine
{
	class Geometry
	{
		public:
			static bool intersect(Ray ray, Sphere sphere);
			static bool intersect(Ray ray, Bounds bounds);
			static bool intersect(Ray ray, Capsule capsule);
			static bool intersect(Sphere sphere, Bounds bounds);
			static bool intersect(Sphere sphere1, Sphere sphere2);
			static bool intersect(Sphere sphere, Capsule capsule);
			static bool intersect(Bounds bounds1, Bounds bounds2);
			static bool intersect(Bounds bounds, Capsule capsule);
	};
}

#endif