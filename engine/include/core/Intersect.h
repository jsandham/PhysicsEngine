#ifndef __INTERSECT_H__
#define __INTERSECT_H__

#include "Ray.h"
#include "Triangle.h"
#include "Plane.h"
#include "Sphere.h"
#include "AABB.h"
#include "Capsule.h"
#include "Frustum.h"

namespace PhysicsEngine
{
	class Intersect
	{
		private:
			static float EPSILON;

		public:
			static bool intersect(Ray ray, Triangle triangle);
			static bool intersect(Ray ray, Plane plane);
			static bool intersect(Ray ray, Sphere sphere);
			static bool intersect(Ray ray, AABB aabb);
			static bool intersect(Ray ray, Capsule capsule);
			static bool intersect(Ray ray, Frustum frustum);
			static bool intersect(Sphere sphere, Sphere sphere2);
			static bool intersect(Sphere sphere, AABB aabb);
			static bool intersect(Sphere sphere, Capsule capsule);
			static bool intersect(Sphere sphere, Frustum frustum);
			static bool intersect(AABB aabb, AABB aabb2);
			static bool intersect(AABB aabb, Capsule capsule);
			static bool intersect(AABB aabb, Frustum frustum);
			static bool intersect(Frustum frustum, Frustum frustum2);
	};
}

#endif