#ifndef __LINE_H__
#define __LINE_H__

#include "Asset.h"

#include "../graphics/GraphicsHandle.h"

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class Line : public Asset // why is this an asset??? Do we even use this class or need to use it?
	{
		public:
			glm::vec3 start;
			glm::vec3 end;

			GraphicsHandle lineVAO;
			GraphicsHandle vertexVBO;

		public:
			Line();
			Line(glm::vec3 start, glm::vec3 end);
			~Line();
	};
}

#endif