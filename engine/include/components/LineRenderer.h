#ifndef __LINERENDERER_H__
#define __LINERENDERER_H__

#include <vector>

#include "Component.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct LineRendererHeader
	{
		Guid componentId;
		Guid entityId;
		glm::vec3 start;
		glm::vec3 end;

		Guid materialId;
	};
#pragma pack(pop)

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

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);
	};

	template <>
	const int ComponentType<LineRenderer>::type = 4;

	template <typename T>
	struct IsLineRenderer { static bool value; };

	template <typename T>
	bool IsLineRenderer<T>::value = false;

	template<>
	bool IsLineRenderer<LineRenderer>::value = true;
	template<>
	bool IsComponent<LineRenderer>::value = true;
}

#endif