#ifndef __BOXCOLLIDER_H__
#define __BOXCOLLIDER_H__

#include "Collider.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class BoxCollider : public Collider
	{
		public:
			Bounds bounds;

		public:
			BoxCollider();
			BoxCollider(Entity *entity);
			~BoxCollider();

			bool intersect(Bounds bounds);
	};
}

#endif