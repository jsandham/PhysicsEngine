#ifndef __DEBUGLINE_H__
#define __DEBUGLINE_H__

#include <vector>

#include "VertexArrayObject.h"
#include "Buffer.h"

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class DebugLine
	{
		private:
			Buffer vertexVBO;
			VertexArrayObject vertexVAO;

			std::vector<float> vertices;

		public:
			DebugLine(glm::vec3 start, glm::vec3 end);
			~DebugLine();

			void draw();
	};
}

#endif