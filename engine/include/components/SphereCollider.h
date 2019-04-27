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

			bool intersect(Bounds bounds);

			std::vector<float> getLines() const;
	};
}

#endif