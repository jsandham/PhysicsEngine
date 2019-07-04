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
		glm::mat4 projection;
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
			glm::mat4 projection;
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
	};
}

#endif