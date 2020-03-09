#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__

#include <vector>

#include "Component.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtx/quaternion.hpp"
#include "../glm/gtc/matrix_transform.hpp"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct TransformHeader
	{
		Guid componentId;
		Guid parentId;
		Guid entityId;
		glm::vec3 position;
		glm::quat rotation;
		glm::vec3 scale;
	};
#pragma pack(pop)

	class Transform : public Component
	{
		public:
			Guid parentId;
			glm::vec3 position;
			glm::quat rotation;
			glm::vec3 scale;

		public:
			Transform();
			Transform(std::vector<char> data);
			~Transform();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
			void deserialize(std::vector<char> data);

			glm::vec3 getEulerAngles() const;
			glm::mat4 getModelMatrix() const;

			void setEulerAngles(glm::vec3 eulerAngles);
	};

	template <>
	const int ComponentType<Transform>::type = 0;

	template <typename T>
	struct IsTransform { static bool value; };

	template <typename T>
	bool IsTransform<T>::value = false;

	template<>
	bool IsTransform<Transform>::value = true;
	template<>
	bool IsComponent<Transform>::value = true;
}

#endif