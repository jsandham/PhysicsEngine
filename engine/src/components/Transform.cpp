#include "../../include/components/Transform.h"

using namespace PhysicsEngine;

Transform::Transform()
{
	mParentId = Guid::INVALID;
	mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	mRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
	mScale = glm::vec3(1.0f, 1.0f, 1.0f);
}

Transform::Transform(const std::vector<char>& data)
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

void Transform::deserialize(const std::vector<char>& data)
{
	const TransformHeader* header = reinterpret_cast<const TransformHeader*>(&data[0]);

	mComponentId = header->mComponentId;
	mParentId = header->mParentId;
	mEntityId = header->mEntityId;
	mPosition = header->mPosition;
	mRotation = header->mRotation;
	mScale = header->mScale;
}

glm::mat4 Transform::getModelMatrix() const
{
	glm::mat4 modelMatrix = glm::translate(glm::mat4(), mPosition);
	modelMatrix *= glm::toMat4(mRotation);
	modelMatrix = glm::scale(modelMatrix, mScale);

	return modelMatrix;
}

glm::vec3 Transform::getForward() const
{
	// a transform with zero rotation has its blue axis pointing in the z direction
	return glm::vec3(glm::rotate(mRotation, glm::vec4(0, 0, 1, 0))); 
}

glm::vec3 Transform::getUp() const
{
	// a transform with zero rotation has its green axis pointing in the y direction
	return glm::vec3(glm::rotate(mRotation, glm::vec4(0, 1, 0, 0)));
}

glm::vec3 Transform::getRight() const
{
	// a transform with zero rotation has its red axis pointing in the x direction
	return glm::vec3(glm::rotate(mRotation, glm::vec4(1, 0, 0, 0)));
}