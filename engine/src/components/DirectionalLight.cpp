#include <iostream>

#include "../../include/components/DirectionalLight.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

DirectionalLight::DirectionalLight()
{
	direction = glm::vec3(1.0f, 2.0f, 0.0f);
	ambient = glm::vec3(0.4f, 0.4f, 0.4f);
	diffuse = glm::vec3(1.0f, 1.0f, 1.0f);
	specular = glm::vec3(1.0f, 1.0f, 1.0f);
}

DirectionalLight::DirectionalLight(std::vector<char> data)
{
	size_t index = sizeof(char);
	index += sizeof(int);
	DirectionalLightHeader* header = reinterpret_cast<DirectionalLightHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	direction = header->direction;
	ambient = header->ambient;
	diffuse = header->diffuse;
	specular = header->specular;
}

DirectionalLight::~DirectionalLight()
{

}