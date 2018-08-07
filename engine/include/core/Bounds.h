#ifndef __BOUNDS_H__
#define __BOUNDS_H__

#include <vector>

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class Bounds
	{
		public:
			glm::vec3 centre;
			glm::vec3 size;

		public:
			Bounds();
			Bounds(glm::vec3 centre, glm::vec3 size);
			~Bounds();

			glm::vec3 getExtents();
			glm::vec3 getMin();
			glm::vec3 getMax();
	};
}

#endif