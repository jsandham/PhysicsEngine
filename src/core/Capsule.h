#ifndef __CAPSULE_H__
#define __CAPSULE_H__

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class Capsule
	{
		public:
			glm::vec3 centre;
			float radius;
			float height;

		public:
			Capsule();
			Capsule(glm::vec3 centre, float radius, float height);
			~Capsule();
	};
}

#endif