#include <iostream>

#include "../../include/components/Transform.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

Transform::Transform()
{
	this->parentId = Guid::INVALID;
	this->position = glm::vec3(0.0f, 0.0f, 0.0f);
	this->rotation = glm::quat(glm::vec3(0.0f, 0.0f, 0.0f));
	this->scale = glm::vec3(1.0f, 1.0f, 1.0f);
}

Transform::Transform(std::vector<char> data)
{
	deserialize(data);
}

Transform::~Transform()
{

}

std::vector<char> Transform::serialize()
{
	TransformHeader header;
	header.componentId = componentId;
	header.parentId = parentId;
	header.entityId = entityId;
	header.position = position;
	header.rotation = rotation;
	header.scale = scale;

	int numberOfBytes = sizeof(TransformHeader);

	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &header, sizeof(TransformHeader));

	return data;
}

void Transform::deserialize(std::vector<char> data)
{
	TransformHeader* header = reinterpret_cast<TransformHeader*>(&data[0]);

	componentId = header->componentId;
	parentId = header->parentId;
	entityId = header->entityId;
	position = header->position;
	rotation = header->rotation;
	scale = header->scale;
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