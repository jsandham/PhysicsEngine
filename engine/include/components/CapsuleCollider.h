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

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			bool intersect(Bounds bounds);
	};

	template <>
	const int ComponentType<CapsuleCollider>::type = 10;

	template <typename T>
	struct IsCapsuleCollider { static bool value; };

	template <typename T>
	bool IsCapsuleCollider<T>::value = false;

	template<>
	bool IsCapsuleCollider<CapsuleCollider>::value = true;
	template<>
	bool IsCollider<CapsuleCollider>::value = true;
	template<>
	bool IsComponent<CapsuleCollider>::value = true;
}

#endif