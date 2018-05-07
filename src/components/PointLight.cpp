#include "PointLight.h"

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

	projection = glm::perspective(glm::radians(90.0f), 1.0f * 1080 / 1080, 0.1f, 25.0f);
	//projection = glm::perspective(90.0f, 1.0f * 1920 / 1080, 0.1f, 100.0f);
}

PointLight::PointLight(Entity *entity)
{
	this->entity = entity;

	constant = 1.0f;
	linear = 0.1f;
	quadratic = 0.032f;
	position = glm::vec3(2.0f, 3.5f, 0.0f);
	ambient = glm::vec3(0.0f, 0.0f, 0.0f);
	diffuse = glm::vec3(1.0f, 1.0f, 1.0f);
	specular = glm::vec3(1.0f, 1.0f, 1.0f);

	projection = glm::perspective(glm::radians(90.0f), 1.0f * 1080 / 1080, 0.1f, 25.0f);
	//projection = glm::perspective(90.0f, 1.0f * 1920 / 1080, 0.1f, 100.0f);
}

PointLight::~PointLight()
{

}