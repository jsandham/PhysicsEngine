#ifndef __CAPSULECOLLIDER_H__
#define __CAPSULECOLLIDER_H__

#include <vector>

#include "Collider.h"

#include "../core/Capsule.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct CapsuleColliderHeader
	{
		Guid componentId;
		Guid entityId;
		Capsule capsule;
	};
#pragma pack(pop)

	class CapsuleCollider : public Collider
	{
		public:
			Capsule capsule;

		public:
			CapsuleCollider();
			CapsuleCollider(std::vector<char> data);
			~CapsuleCollider();

			bool intersect(Bounds bounds);
	};
}

#endif