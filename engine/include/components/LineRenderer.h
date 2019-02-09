#ifndef __LINERENDERER_H__
#define __LINERENDERER_H__

#include <vector>

#include "Component.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
// #pragma pack(push, 1)
	struct LineRendererData
	{
		Guid componentId;
		Guid entityId;
		glm::vec3 start;
		glm::vec3 end;

		Guid materialId;

	};
// #pragma pack(pop)

	class LineRenderer : public Component
	{
		public:
			glm::vec3 start;
			glm::vec3 end;

			Guid materialId;

		public:
			LineRenderer();
			LineRenderer(std::vector<char> data);
			~LineRenderer();

			void* operator new(size_t size);
			void operator delete(void*);

			void load(LineRendererData data);
	};
}

#endif