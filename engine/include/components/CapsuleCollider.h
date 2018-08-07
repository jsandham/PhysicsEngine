#ifndef __CAPSULECOLLIDER_H__
#define __CAPSULECOLLIDER_H__

#include "Collider.h"

#include "../core/Capsule.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class CapsuleCollider : public Collider
	{
		public:
			Capsule capsule;

		public:
			CapsuleCollider();
			~CapsuleCollider();

			bool intersect(Bounds bounds);
	};
}

#endif