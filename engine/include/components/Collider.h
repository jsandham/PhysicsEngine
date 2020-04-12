#ifndef __COLLIDER_H__
#define __COLLIDER_H__

#include "Component.h"

#include "../core/Ray.h"
#include "../core/AABB.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class Collider : public Component
	{
		public:
			Collider();
			virtual ~Collider() = 0;

			virtual bool intersect(AABB aabb) const = 0;
	};

	template <typename T>
	struct IsCollider { static const bool value; };

	template <typename T>
	const bool IsCollider<T>::value = false;

	template<>
	const bool IsCollider<Collider>::value = true;
	template<>
	const bool IsComponent<Collider>::value = true;
}

#endif