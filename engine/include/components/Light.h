#ifndef __LIGHT_H__
#define __LIGHT_H__

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"

#include "Component.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct LightHeader
	{
		Guid componentId;
		Guid entityId;
		glm::vec3 position;
		glm::vec3 direction;
		glm::vec3 ambient;
		glm::vec3 diffuse;
		glm::vec3 specular;
		float constant;
		float linear;
		float quadratic;
		float cutOff;
		float outerCutOff;
		int lightType;
		int shadowType;
	};
#pragma pack(pop)
	
	enum class LightType
	{
		Directional,
		Spot,
		Point,
		None
	};

	enum class ShadowType
	{
		Hard,
		Soft,
		None
	};

	class Light : public Component
	{
		public:
			glm::vec3 position;
			glm::vec3 direction;
			glm::vec3 ambient;
			glm::vec3 diffuse;
			glm::vec3 specular;
			float constant;
			float linear;
			float quadratic;
			float cutOff;
			float outerCutOff;
			LightType lightType;
			ShadowType shadowType;

		public:
			Light();
			Light(std::vector<char> data);
			~Light();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
			void deserialize(std::vector<char> data);

			glm::mat4 getProjMatrix() const;
	};

	template <>
	const int ComponentType<Light>::type = 5;

	template <typename T>
	struct IsLight { static bool value; };

	template <typename T>
	bool IsLight<T>::value = false;

	template<>
	bool IsLight<Light>::value = true;
	template<>
	bool IsComponent<Light>::value = true;
}

#endif