#ifndef __SPHERECOLLIDER_H__
#define __SPHERECOLLIDER_H__

#include <vector>

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

#include "../core/Sphere.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct SphereColliderHeader
	{
		Guid mComponentId;
		Guid mEntityId;
		Sphere mSphere;
	};
#pragma pack(pop)

	class SphereCollider : public Collider
	{
		public:
			Sphere mSphere;

		public:
			SphereCollider();
			SphereCollider(std::vector<char> data);
			~SphereCollider();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
			void deserialize(std::vector<char> data);

			bool intersect(Bounds bounds) const;

			std::vector<float> getLines() const;
	};

	template <>
	const int ComponentType<SphereCollider>::type = 9;

	template <typename T>
	struct IsSphereCollider { static const bool value; };

	template <typename T>
	const bool IsSphereCollider<T>::value = false;

	template<>
	const bool IsSphereCollider<SphereCollider>::value = true;
	template<>
	const bool IsCollider<SphereCollider>::value = true;
	template<>
	const bool IsComponent<SphereCollider>::value = true;
}

#endif