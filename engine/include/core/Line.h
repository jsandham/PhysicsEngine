#ifndef __LINE_H__
#define __LINE_H__

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class Line
	{
		public:
			glm::vec3 start;
			glm::vec3 end;

		public:
			Line();
			Line(glm::vec3 start, glm::vec3 end);
			~Line();
	};
}

#endif