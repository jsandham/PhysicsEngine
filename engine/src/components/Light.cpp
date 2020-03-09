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
	lightType = LightType::Directional;
	shadowType = ShadowType::Hard;
}

Light::Light(std::vector<char> data)
{
	deserialize(data);
}

Light::~Light()
{

}

std::vector<char> Light::serialize() const
{
	return serialize(componentId, entityId);
}

std::vector<char> Light::serialize(Guid componentId, Guid entityId) const
{
	LightHeader header;
	header.componentId = componentId;
	header.entityId = entityId;
	header.position = position;
	header.direction = direction;
	header.ambient = ambient;
	header.diffuse = diffuse;
	header.specular = specular;
	header.constant = constant;
	header.linear = linear;
	header.quadratic = quadratic;
	header.cutOff = cutOff;
	header.outerCutOff = outerCutOff;
	header.lightType = static_cast<int>(lightType);
	header.shadowType = static_cast<int>(shadowType);

	std::vector<char> data(sizeof(LightHeader));

	memcpy(&data[0], &header, sizeof(LightHeader));

	return data;
}

void Light::deserialize(std::vector<char> data)
{
	LightHeader* header = reinterpret_cast<LightHeader*>(&data[0]);

	componentId = header->componentId;
	entityId = header->entityId;

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

glm::mat4 Light::getProjMatrix() const
{
	return glm::perspective(2.0f * glm::radians(outerCutOff), 1.0f * 1024 / 1024, 0.1f, 12.0f);
}