#include <iostream>
#include "Transform.h"

#include "../core/Log.h"

using namespace PhysicsEngine;

Transform::Transform()
{
	this->position = glm::vec3(0.0f, 0.0f, 0.0f);
	this->eulerAngles = glm::vec3(0.0f, 0.0f, 0.0f);
	this->rotation = glm::quat(eulerAngles);
	this->scale = glm::vec3(1.0f, 1.0f, 1.0f);

	translateMatrix = glm::translate(glm::mat4(1.0f), position);
	scaleMatrix = glm::scale(glm::mat4(1.0f), scale);
	rotationMatrix = glm::toMat4(rotation);

	modelMatrix = translateMatrix * rotationMatrix * scaleMatrix;
}

Transform::Transform(Entity *entity)
{
	this->entity = entity;

	this->position = glm::vec3(0.0f, 0.0f, 0.0f);
	this->eulerAngles = glm::vec3(0.0f, 0.0f, 0.0f);
	this->rotation = glm::quat(eulerAngles);
	this->scale = glm::vec3(1.0f, 1.0f, 1.0f);

	translateMatrix = glm::translate(glm::mat4(1.0f), position);
	scaleMatrix = glm::scale(glm::mat4(1.0f), scale);
	rotationMatrix = glm::toMat4(rotation);

	modelMatrix = translateMatrix * rotationMatrix * scaleMatrix;
}

Transform::~Transform()
{

}

glm::vec3 Transform::getEulerAngles()
{
	return eulerAngles;
}

glm::mat4 Transform::getModelMatrix()
{
	translateMatrix = glm::translate(glm::mat4(1.0f), position);
	scaleMatrix = glm::scale(glm::mat4(1.0f), scale);
	rotationMatrix = glm::toMat4(rotation);

	modelMatrix = translateMatrix * rotationMatrix * scaleMatrix;

	return modelMatrix;
}

void Transform::setEulerAngles(glm::vec3 eulerAngles)
{
	this->eulerAngles = eulerAngles;
	this->rotation = glm::quat(eulerAngles);

	rotationMatrix = glm::toMat4(rotation);
}