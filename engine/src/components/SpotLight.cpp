#include "../../include/components/SpotLight.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

SpotLight::SpotLight()
{
	constant = 1.0f;
	linear = 0.1f;
	quadratic = 0.032f;
	cutOff = glm::cos(glm::radians(12.5f));
	outerCutOff = glm::cos(glm::radians(15.0f));
	position = glm::vec3(0.0f, 1.0f, 0.0f);
	direction = glm::vec3(0.0f, -2.0f, 0.0f);
	ambient = glm::vec3(0.8f, 0.8f, 0.8f);
	diffuse = glm::vec3(1.0f, 1.0f, 1.0f);
	specular = glm::vec3(1.0f, 1.0f, 1.0f);

	projection = glm::perspective(glm::radians(45.0f), 1.0f * 640 / 480, 0.1f, 100.0f);
}

SpotLight::SpotLight(unsigned char* data)
{
	
}

SpotLight::~SpotLight()
{

}

void* SpotLight::operator new(size_t size)
{
	return getAllocator<SpotLight>().allocate();
}

void SpotLight::operator delete(void*)
{

}

void SpotLight::load(SpotLightData data)
{
	entityId = data.entityId;
	componentId = data.componentId;

	constant = data.constant;
	linear = data.linear;
	quadratic = data.quadratic;
	cutOff = data.cutOff;
	outerCutOff = data.outerCutOff;
	position = data.position;
	direction = data.direction;
	ambient = data.ambient;
	diffuse = data.diffuse;
	specular = data.specular;
}