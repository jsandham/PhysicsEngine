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
		Guid mComponentId;
		Guid mEntityId;
		glm::vec3 mStart;
		glm::vec3 mEnd;

		Guid mMaterialId;
	};
#pragma pack(pop)

	class LineRenderer : public Component
	{
		public:
			glm::vec3 mStart;
			glm::vec3 mEnd;

			Guid mMaterialId;

		public:
			LineRenderer();
			LineRenderer(std::vector<char> data);
			~LineRenderer();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
			void deserialize(std::vector<char> data);
	};

	template <>
	const int ComponentType<LineRenderer>::type = 4;

	template <typename T>
	struct IsLineRenderer { static const bool value; };

	template <typename T>
	const bool IsLineRenderer<T>::value = false;

	template<>
	const bool IsLineRenderer<LineRenderer>::value = true;
	template<>
	const bool IsComponent<LineRenderer>::value = true;
}

#endif