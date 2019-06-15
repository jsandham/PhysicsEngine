#include "../../include/components/PointLight.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

PointLight::PointLight()
{
	constant = 1.0f;
	linear = 0.1f;
	quadratic = 0.032f;
	position = glm::vec3(2.0f, 3.5f, 0.0f);
	ambient = glm::vec3(0.0f, 0.0f, 0.0f);
	diffuse = glm::vec3(1.0f, 1.0f, 1.0f);
	specular = glm::vec3(1.0f, 1.0f, 1.0f);

	projection = glm::perspective(glm::radians(90.0f), 1.0f * 1024 / 1024, 0.1f, 250.0f);

	lightType = LightType::Point;
	shadowType = ShadowType::Hard;
}

PointLight::PointLight(std::vector<char> data)
{
	size_t index = sizeof(char);
	index += sizeof(int);
	PointLightHeader* header = reinterpret_cast<PointLightHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	constant = header->constant;
	linear = header->linear;
	quadratic = header->quadratic;
	position = header->position;
	ambient = header->ambient;
	diffuse = header->diffuse;
	specular = header->specular;
	projection = header->projection;

	lightType = LightType::Point;
	shadowType = ShadowType::Hard;
}

PointLight::~PointLight()
{

}