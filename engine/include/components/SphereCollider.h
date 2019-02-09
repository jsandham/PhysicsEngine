#ifndef __SPHERECOLLIDER_H__
#define __SPHERECOLLIDER_H__

#include <vector>

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

#include "../core/Sphere.h"

namespace PhysicsEngine
{
// #pragma pack(push, 1)
	struct SphereColliderData
	{
		Guid componentId;
		Guid entityId;
		Sphere sphere;
	};
// #pragma pack(pop)

	class SphereCollider : public Collider
	{
		public:
			Sphere sphere;

		public:
			SphereCollider();
			SphereCollider(std::vector<char> data);
			~SphereCollider();

			void* operator new(size_t size);
			void operator delete(void*);

			void load(SphereColliderData data);

			bool intersect(Bounds bounds);
	};
}

#endif