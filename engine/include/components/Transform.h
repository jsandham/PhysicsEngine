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
		Guid mComponentId;
		Guid mParentId;
		Guid mEntityId;
		glm::vec3 mPosition;
		glm::quat mRotation;
		glm::vec3 mScale;
	};
#pragma pack(pop)

	class Transform : public Component
	{
		public:
			Guid mParentId;
			glm::vec3 mPosition;
			glm::quat mRotation;
			glm::vec3 mScale;

		public:
			Transform();
			Transform(const std::vector<char>& data);
			~Transform();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
			void deserialize(const std::vector<char>& data);

			glm::mat4 getModelMatrix() const;
			glm::vec3 getForward() const;
			glm::vec3 getUp() const;
			glm::vec3 getRight() const;
	};

	template <>
	const int ComponentType<Transform>::type = 0;

	template <typename T>
	struct IsTransform { static const bool value; };

	template <typename T>
	const bool IsTransform<T>::value = false;

	template<>
	const bool IsTransform<Transform>::value = true;
	template<>
	const bool IsComponent<Transform>::value = true;
	template<>
	const bool IsComponentInternal<Transform>::value = true;

}

#endif