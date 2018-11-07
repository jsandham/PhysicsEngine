#ifndef __LINE_H__
#define __LINE_H__

#include "../graphics/GLHandle.h"

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class Line
	{
		public:
			glm::vec3 start;
			glm::vec3 end;

			GLHandle lineVAO;
			GLHandle vertexVBO;

		public:
			Line();
			Line(glm::vec3 start, glm::vec3 end);
			~Line();
	};
}

#endif