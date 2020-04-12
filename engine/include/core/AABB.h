#ifndef __AABB_H__
#define __AABB_H__

#include <vector>

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class AABB
	{
		public:
			glm::vec3 mCentre;
			glm::vec3 mSize;

		public:
			AABB();
			AABB(glm::vec3 centre, glm::vec3 size);
			~AABB();

			glm::vec3 getExtents() const;
			glm::vec3 getMin() const;
			glm::vec3 getMax() const;
	};
}

#endif