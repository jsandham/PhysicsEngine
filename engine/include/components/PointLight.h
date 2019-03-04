#ifndef __POINTLIGHT_H__
#define __POINTLIGHT_H__

#include <vector>

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"

#include "Component.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct PointLightHeader
	{
		Guid componentId;
		Guid entityId;
		float constant;
		float linear;
		float quadratic;
		glm::vec3 position;
		glm::vec3 ambient;
		glm::vec3 diffuse;
		glm::vec3 specular;
		glm::mat4 projection;
	};
#pragma pack(pop)

	class PointLight : public Component
	{
		public:
			float constant;
			float linear;
			float quadratic;
			glm::vec3 position;
			glm::vec3 ambient;
			glm::vec3 diffuse;
			glm::vec3 specular;
			glm::mat4 projection;

		public:
			PointLight();
			PointLight(std::vector<char> data);
			~PointLight();
	};
}

#endif