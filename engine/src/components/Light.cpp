#include "../../include/components/Light.h"

using namespace PhysicsEngine;

Light::Light()
{
	componentId = Guid::INVALID;
	entityId = Guid::INVALID;

	position = glm::vec3(0.0f, 1.0f, 0.0f);
	direction = glm::vec3(1.0f, 2.0f, 0.0f);
	ambient = glm::vec3(0.4f, 0.4f, 0.4f);
	diffuse = glm::vec3(1.0f, 1.0f, 1.0f);
	specular = glm::vec3(1.0f, 1.0f, 1.0f);
	constant = 1.0f;
	linear = 0.1f;
	quadratic = 0.032f;
	cutOff = glm::cos(glm::radians(12.5f));
	outerCutOff = glm::cos(glm::radians(15.0f));

	projection = glm::perspective(2.0f * glm::radians(outerCutOff), 1.0f * 1024 / 1024, 0.1f, 12.0f);

	lightType = LightType::Directional;
	shadowType = ShadowType::Hard;
}

Light::Light(std::vector<char> data)
{
	size_t index = sizeof(char);
	index += sizeof(int);
	LightHeader* header = reinterpret_cast<LightHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;

	projection = header->projection;
	position = header->position;
	direction = header->direction;
	ambient = header->ambient;
	diffuse = header->diffuse;
	specular = header->specular;

	constant = header->constant;
	linear = header->linear;
	quadratic = header->quadratic;
	cutOff = glm::cos(glm::radians(header->cutOff));
	outerCutOff = glm::cos(glm::radians(header->outerCutOff));

	lightType = static_cast<LightType>(header->lightType);
	shadowType = static_cast<ShadowType>(header->shadowType);
}

Light::~Light()
{

}


