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
	
}

DirectionalLight::~DirectionalLight()
{

}

void* DirectionalLight::operator new(size_t size)
{
	return getAllocator<DirectionalLight>().allocate();
}

void DirectionalLight::operator delete(void*)
{

}

void DirectionalLight::load(DirectionalLightData data)
{
	entityId = data.entityId;
	componentId = data.componentId;

	direction = data.direction;
	ambient = data.ambient;
	diffuse = data.diffuse;
	specular = data.specular;
}