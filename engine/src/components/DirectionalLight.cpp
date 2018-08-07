#include <iostream>

#include "../../include/components/DirectionalLight.h"

using namespace PhysicsEngine;

DirectionalLight::DirectionalLight()
{
	direction = glm::vec3(1.0f, 2.0f, 0.0f);
	ambient = glm::vec3(0.4f, 0.4f, 0.4f);
	diffuse = glm::vec3(1.0f, 1.0f, 1.0f);
	specular = glm::vec3(1.0f, 1.0f, 1.0f);
}

DirectionalLight::~DirectionalLight()
{

}