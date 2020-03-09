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

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
			void deserialize(std::vector<char> data);

			bool intersect(Bounds bounds) const;
	};

	template <>
	const int ComponentType<MeshCollider>::type = 15;

	template <typename T>
	struct IsMeshCollider { static bool value; };

	template <typename T>
	bool IsMeshCollider<T>::value = false;

	template<>
	bool IsMeshCollider<MeshCollider>::value = true;
	template<>
	bool IsComponent<MeshCollider>::value = true;
}

#endif