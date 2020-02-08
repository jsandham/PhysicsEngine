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
		Guid componentId;
		Guid entityId;
		Sphere sphere;
	};
#pragma pack(pop)

	class SphereCollider : public Collider
	{
		public:
			Sphere sphere;

		public:
			SphereCollider();
			SphereCollider(std::vector<char> data);
			~SphereCollider();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			bool intersect(Bounds bounds);

			std::vector<float> getLines() const;
	};

	template <>
	const int ComponentType<SphereCollider>::type = 9;

	template <typename T>
	struct IsSphereCollider { static bool value; };

	template <typename T>
	bool IsSphereCollider<T>::value = false;

	template<>
	bool IsSphereCollider<SphereCollider>::value = true;
	template<>
	bool IsCollider<SphereCollider>::value = true;
	template<>
	bool IsComponent<SphereCollider>::value = true;
}

#endif