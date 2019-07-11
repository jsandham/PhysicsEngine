#ifndef __MESHCOLLIDER_H__
#define __MESHCOLLIDER_H__

#include <vector>

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct MeshColliderHeader
	{
		Guid componentId;
		Guid entityId;
		Guid meshId;
	};
#pragma pack(pop)

	class MeshCollider : public Collider
	{
		public:
			Guid meshId;

		public:
			MeshCollider();
			MeshCollider(std::vector<char> data);
			~MeshCollider();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			bool intersect(Bounds bounds);
	};
}

#endif