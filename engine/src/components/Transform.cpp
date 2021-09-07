#include "../../include/components/Transform.h"

#include "glm/gtx/matrix_decompose.hpp"

using namespace PhysicsEngine;

#define GLM_ENABLE_EXPERIMENTAL

Transform::Transform(World* world) : Component(world)
{
    mParentId = Guid::INVALID;
    mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
    mRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    mScale = glm::vec3(1.0f, 1.0f, 1.0f);
}

Transform::Transform(World* world, Guid id) : Component(world, id)
{
    mParentId = Guid::INVALID;
    mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
    mRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    mScale = glm::vec3(1.0f, 1.0f, 1.0f);
}

Transform::~Transform()
{
}

void Transform::serialize(YAML::Node &out) const
{
    Component::serialize(out);

    out["parentId"] = mParentId;
    out["position"] = mPosition;
    out["rotation"] = mRotation;
    out["scale"] = mScale;
}

void Transform::deserialize(const YAML::Node &in)
{
    Component::deserialize(in);

    mParentId = YAML::getValue<Guid>(in, "parentId");
    mPosition = YAML::getValue<glm::vec3>(in, "position");
    mRotation = YAML::getValue<glm::quat>(in, "rotation");
    mScale = YAML::getValue<glm::vec3>(in, "scale");
}

int Transform::getType() const
{
    return PhysicsEngine::TRANSFORM_TYPE;
}

std::string Transform::getObjectName() const
{
    return PhysicsEngine::TRANSFORM_NAME;
}

glm::mat4 Transform::getModelMatrix() const
{
    glm::mat4 modelMatrix = glm::translate(glm::mat4(1.0f), mPosition);
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

bool Transform::decompose(const glm::mat4& model, glm::vec3& translation, glm::quat& rotation, glm::vec3& scale)
{
	glm::vec3 skew;
	glm::vec4 perspective;
	return glm::decompose(model, scale, rotation, translation, skew, perspective);
}

void Transform::v3Scale(glm::vec3& v, float desiredLength)
{
	float len = glm::length(v);
	if (len != 0)
	{
		float l = desiredLength / len;
		v[0] *= l;
		v[1] *= l;
		v[2] *= l;
	}
}