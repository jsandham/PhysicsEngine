#ifndef __SPHERECOLLIDER_H__
#define __SPHERECOLLIDER_H__

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

#include "../core/Sphere.h"

namespace PhysicsEngine
{
	class SphereCollider : public Collider
	{
		public:
			Sphere sphere;

		public:
			SphereCollider();
			~SphereCollider();

			bool intersect(Bounds bounds);
	};
}

#endif