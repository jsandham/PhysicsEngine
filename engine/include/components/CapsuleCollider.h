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
		Guid mComponentId;
		Guid mEntityId;
		Capsule mCapsule;
	};
#pragma pack(pop)

	class CapsuleCollider : public Collider
	{
		public:
			Capsule mCapsule;

		public:
			CapsuleCollider();
			CapsuleCollider(const std::vector<char>& data);
			~CapsuleCollider();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
			void deserialize(const std::vector<char>& data);

			bool intersect(AABB aabb) const;
	};

	template <>
	const int ComponentType<CapsuleCollider>::type = 10;

	template <typename T>
	struct IsCapsuleCollider { static const bool value; };

	template <typename T>
	const bool IsCapsuleCollider<T>::value = false;

	template<>
	const bool IsCapsuleCollider<CapsuleCollider>::value = true;
	template<>
	const bool IsCollider<CapsuleCollider>::value = true;
	template<>
	const bool IsComponent<CapsuleCollider>::value = true;
	template<>
	const bool IsComponentInternal<CapsuleCollider>::value = true;
}

#endif