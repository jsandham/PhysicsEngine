#ifndef __COLLIDER_H__
#define __COLLIDER_H__

#include "Component.h"

#include "../core/Ray.h"
#include "../core/Bounds.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class Collider : public Component
	{
		public:
			Collider();
			virtual ~Collider() = 0;

			virtual bool intersect(Bounds bounds) = 0;
	};
}

#endif