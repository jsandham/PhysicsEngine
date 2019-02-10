#ifndef __BOXCOLLIDER_H__
#define __BOXCOLLIDER_H__

#include <vector>

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct BoxColliderHeader
	{
		Guid componentId;
		Guid entityId;
		Bounds bounds;
	};
#pragma pack(pop)

	class BoxCollider : public Collider
	{
		public:
			Bounds bounds;

		public:
			BoxCollider();
			BoxCollider(std::vector<char> data);
			~BoxCollider();

			void* operator new(size_t size);
			void operator delete(void*);

			bool intersect(Bounds bounds);
	};
}

#endif