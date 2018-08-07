#include <iostream>

#include "../../include/components/Transform.h"

using namespace PhysicsEngine;

Transform::Transform()
{
	this->position = glm::vec3(0.0f, 0.0f, 0.0f);
	this->rotation = glm::quat(glm::vec3(0.0f, 0.0f, 0.0f));
	this->scale = glm::vec3(1.0f, 1.0f, 1.0f);
}

Transform::~Transform()
{

}

glm::vec3 Transform::getEulerAngles()
{
	return glm::eulerAngles(rotation);
}

glm::mat4 Transform::getModelMatrix()
{
	return glm::translate(glm::mat4(1.0f), position) * glm::toMat4(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

void Transform::setEulerAngles(glm::vec3 eulerAngles)
{
	this->rotation = glm::quat(eulerAngles);
}