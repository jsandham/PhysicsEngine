#include <iostream>

#include "../../include/components/Transform.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

Transform::Transform()
{
	this->position = glm::vec3(0.0f, 0.0f, 0.0f);
	this->rotation = glm::quat(glm::vec3(0.0f, 0.0f, 0.0f));
	this->scale = glm::vec3(1.0f, 1.0f, 1.0f);
}

Transform::Transform(unsigned char* data)
{
	
}

Transform::~Transform()
{

}

void* Transform::operator new(size_t size)
{
	return getAllocator<Transform>().allocate();
}

void Transform::operator delete(void*)
{

}

void Transform::load(TransformData data)
{
	entityId = data.entityId;
	componentId = data.componentId;

	position = data.position;
	rotation = data.rotation;
	scale = data.scale;
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