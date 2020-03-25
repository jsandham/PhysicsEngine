#include "../../include/components/Transform.h"

using namespace PhysicsEngine;

Transform::Transform()
{
	mParentId = Guid::INVALID;
	mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	mRotation = glm::quat(glm::vec3(0.0f, 0.0f, 0.0f));
	mScale = glm::vec3(1.0f, 1.0f, 1.0f);
}

Transform::Transform(std::vector<char> data)
{
	deserialize(data);
}

Transform::~Transform()
{

}

std::vector<char> Transform::serialize() const
{
	return serialize(mComponentId, mEntityId);
}

std::vector<char> Transform::serialize(Guid componentId, Guid entityId) const
{
	TransformHeader header;
	header.mComponentId = componentId;
	header.mParentId = mParentId;
	header.mEntityId = entityId;
	header.mPosition = mPosition;
	header.mRotation = mRotation;
	header.mScale = mScale;

	std::vector<char> data(sizeof(TransformHeader));

	memcpy(&data[0], &header, sizeof(TransformHeader));

	return data;
}

void Transform::deserialize(std::vector<char> data)
{
	TransformHeader* header = reinterpret_cast<TransformHeader*>(&data[0]);

	mComponentId = header->mComponentId;
	mParentId = header->mParentId;
	mEntityId = header->mEntityId;
	mPosition = header->mPosition;
	mRotation = header->mRotation;
	mScale = header->mScale;
}

glm::vec3 Transform::getEulerAngles() const
{
	return glm::eulerAngles(mRotation);
}

glm::mat4 Transform::getModelMatrix() const
{
	glm::mat4 modelMatrix = glm::translate(glm::mat4(), mPosition);
	modelMatrix *= glm::toMat4(mRotation);
	modelMatrix = glm::scale(modelMatrix, mScale);

	return modelMatrix;
	//return glm::translate(glm::mat4(1.0f), position) * glm::toMat4(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

void Transform::setEulerAngles(glm::vec3 eulerAngles)
{
	mRotation = glm::quat(eulerAngles);
}