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

glm::vec3 Transform::getPosition()
{
	return position;
}

glm::vec3 Transform::getEulerAngles()
{
	return eulerAngles;
}

glm::quat Transform::getRotation()
{
	return rotation;
}

glm::vec3 Transform::getScale()
{
	return scale;
}

glm::mat4 Transform::getModelMatrix()
{
	modelMatrix = translateMatrix * rotationMatrix * scaleMatrix;

	return modelMatrix;
}

void Transform::setPosition(glm::vec3 position)
{
	this->position = position;

	translateMatrix = glm::translate(glm::mat4(1.0f), position);
}

void Transform::setEulerAngles(glm::vec3 eulerAngles)
{
	this->eulerAngles = eulerAngles;
	this->rotation = glm::quat(eulerAngles);

	rotationMatrix = glm::toMat4(rotation);
}

void Transform::setRotation(glm::quat rotation)
{
	this->rotation = rotation;
	this->eulerAngles = glm::eulerAngles(rotation);

	rotationMatrix = glm::toMat4(rotation);
}

void Transform::setScale(glm::vec3 scale)
{
	if (scale.x*scale.x < 0.00000001f || scale.y*scale.y < 0.00000001f || scale.z*scale.z < 0.00000001f){
		Log::Warn("Transform: scale vector cannot have zero components. Scale not set");
		return;
	}

	this->scale = scale;

	scaleMatrix = glm::scale(glm::mat4(1.0f), scale);
}