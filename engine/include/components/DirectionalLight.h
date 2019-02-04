#ifndef __DIRECTIONALLIGHT_H__
#define __DIRECTIONALLIGHT_H__

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

#include "Component.h"

namespace PhysicsEngine
{
// #pragma pack(push, 1)
	struct DirectionalLightData
	{
		Guid componentId;
		Guid entityId;
		glm::vec3 direction;
		glm::vec3 ambient;
		glm::vec3 diffuse;
		glm::vec3 specular;
	};
// #pragma pack(pop)

	class DirectionalLight : public Component
	{
		public:
			glm::vec3 direction;
			glm::vec3 ambient;
			glm::vec3 diffuse;
			glm::vec3 specular;

		public:
			DirectionalLight();
			DirectionalLight(unsigned char* data);
			~DirectionalLight();

			void* operator new(size_t size);
			void operator delete(void*);

			void load(DirectionalLightData data);
	};
}

#endif