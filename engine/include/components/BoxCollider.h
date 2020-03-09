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

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
			void deserialize(std::vector<char> data);

			bool intersect(Bounds bounds) const;

			std::vector<float> getLines() const;
	};

	template <>
	const int ComponentType<BoxCollider>::type = 8;

	template <typename T>
	struct IsBoxCollider { static bool value; };

	template <typename T>
	bool IsBoxCollider<T>::value = false;

	template<>
	bool IsBoxCollider<BoxCollider>::value = true;
	template<>
	bool IsCollider<BoxCollider>::value = true;
	template<>
	bool IsComponent<BoxCollider>::value = true;
}

#endif