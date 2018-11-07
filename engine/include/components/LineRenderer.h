#ifndef __LINERENDERER_H__
#define __LINERENDERER_H__

#include "Component.h"
#include "../core/Guid.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class LineRenderer : public Component
	{
		public:
			glm::vec3 start;
			glm::vec3 end;

			Guid materialId;

		public:
			LineRenderer();
			~LineRenderer();
	};
}

#endif