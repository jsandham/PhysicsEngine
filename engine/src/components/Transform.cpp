#include "../../include/components/Transform.h"

#include "../../include/core/Serialization.h"

using namespace PhysicsEngine;

Transform::Transform() : Component()
{
    mParentId = Guid::INVALID;
    mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
    mRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    mScale = glm::vec3(1.0f, 1.0f, 1.0f);
}

Transform::Transform(Guid id) : Component(id)
{
    mParentId = Guid::INVALID;
    mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
    mRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    mScale = glm::vec3(1.0f, 1.0f, 1.0f);
}

Transform::~Transform()
{
}

void Transform::serialize(std::ostream &out) const
{
    Component::serialize(out);

    PhysicsEngine::write<Guid>(out, mParentId);
    PhysicsEngine::write<glm::vec3>(out, mPosition);
    PhysicsEngine::write<glm::quat>(out, mRotation);
    PhysicsEngine::write<glm::vec3>(out, mScale);
}

void Transform::deserialize(std::istream &in)
{
    Component::deserialize(in);

    PhysicsEngine::read<Guid>(in, mParentId);
    PhysicsEngine::read<glm::vec3>(in, mPosition);
    PhysicsEngine::read<glm::quat>(in, mRotation);
    PhysicsEngine::read<glm::vec3>(in, mScale);
}

void Transform::serialize(YAML::Node& out) const
{
    Component::serialize(out);

    out["parentId"] = mParentId;
    out["position"] = mPosition;
    out["rotation"] = mRotation;
    out["scale"] = mScale;
}

void Transform::deserialize(const YAML::Node& in)
{
    Component::deserialize(in);

    mParentId = in["parentId"].as<Guid>();
    mPosition = in["position"].as<glm::vec3>();
    mRotation = in["rotation"].as<glm::quat>();
    mScale = in["scale"].as<glm::vec3>();
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