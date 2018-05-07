#ifndef __DIRECTIONALLIGHT_H__
#define __DIRECTIONALLIGHT_H__

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

#include "Component.h"

namespace PhysicsEngine
{
	class DirectionalLight : public Component
	{
		public:
			glm::vec3 direction;
			glm::vec3 ambient;
			glm::vec3 diffuse;
			glm::vec3 specular;

		public:
			DirectionalLight();
			DirectionalLight(Entity *entity);
			~DirectionalLight();
	};
}

#endif