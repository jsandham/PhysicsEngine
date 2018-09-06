#ifndef __SPOTLIGHT_H__
#define __SPOTLIGHT_H__

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"

#include "Component.h"

namespace PhysicsEngine
{
	class SpotLight : public Component
	{
		public:
			float constant;
			float linear;
			float quadratic;
			float cutOff;
			float outerCutOff;
			glm::vec3 position;
			glm::vec3 direction;
			glm::vec3 ambient;
			glm::vec3 diffuse;
			glm::vec3 specular;
			glm::mat4 projection;

		public:
			SpotLight();
			~SpotLight();
	};
}

#endif