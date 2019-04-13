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

			glm::vec3 getExtents() const;
			glm::vec3 getMin() const;
			glm::vec3 getMax() const;
	};
}

#endif