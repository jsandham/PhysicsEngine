#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class Triangle
	{
		public:
			glm::vec3 v1;
			glm::vec3 v2;
			glm::vec3 v3;

		public:
			Triangle();
			Triangle(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3);
			~Triangle();
	};
}


#endif