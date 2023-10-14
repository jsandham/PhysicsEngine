#include "../../include/components/Transform.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/World.h"
#include "../../include/core/GLM.h"

#include "glm/gtx/matrix_decompose.hpp"

using namespace PhysicsEngine;

#define GLM_ENABLE_EXPERIMENTAL

TransformData::TransformData()
    : mPosition(glm::vec3(0.0f, 0.0f, 0.0f)), mRotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)),
      mScale(glm::vec3(1.0f, 1.0f, 1.0f))
{
}

void TransformData::serialize(YAML::Node &out) const
{
    out["position"] = mPosition; // mPosition;
    out["rotation"] = mRotation; // mRotation;
    out["scale"] = mScale;       // mScale;
}

void TransformData::deserialize(const YAML::Node &in)
{
    mPosition = YAML::getValue<glm::vec3>(in, "position");
    mRotation = YAML::getValue<glm::quat>(in, "rotation");
    mScale = YAML::getValue<glm::vec3>(in, "scale");
}

glm::mat4 TransformData::getModelMatrix() const
{
    //glm::mat4 modelMatrix = glm::translate(glm::mat4(1.0f), mPosition);
    //modelMatrix *= glm::toMat4(mRotation);
    //modelMatrix = glm::scale(modelMatrix, mScale);
    
    // M = T * R * S
    // Rxx*Sx Ryx*Sy Rzx*Sz Tx
    // Rxy*Sx Ryy*Sy Rzy*Sz Ty
    // Rxz*Sx Ryz*Sy Rzz*Sz Tz
    //      0      0      0 1
    glm::mat4 modelMatrix = glm::toMat4(mRotation);
    modelMatrix[3].x = mPosition.x;
    modelMatrix[3].y = mPosition.y;
    modelMatrix[3].z = mPosition.z;

    modelMatrix[0] *= mScale.x;
    modelMatrix[1] *= mScale.y;
    modelMatrix[2] *= mScale.z;

    return modelMatrix;
}

glm::vec3 TransformData::getForward() const
{
    // a transform with zero rotation has its blue axis pointing in the +z direction
    return glm::vec3(glm::rotate(mRotation, glm::vec4(0, 0, 1, 0)));
}

glm::vec3 TransformData::getUp() const
{
    // a transform with zero rotation has its green axis pointing in the +y direction
    return glm::vec3(glm::rotate(mRotation, glm::vec4(0, 1, 0, 0)));
}

glm::vec3 TransformData::getRight() const
{
    // a transform with zero rotation has its red axis pointing in the +x direction
    return glm::vec3(glm::rotate(mRotation, glm::vec4(1, 0, 0, 0)));
}


Transform::Transform(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
}

Transform::Transform(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
}

Transform::~Transform()
{
}

void Transform::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["entityId"] = mEntityGuid;
}

void Transform::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

    mEntityGuid = YAML::getValue<Guid>(in, "entityId");
}

int Transform::getType() const
{
    return PhysicsEngine::TRANSFORM_TYPE;
}

std::string Transform::getObjectName() const
{
    return PhysicsEngine::TRANSFORM_NAME;
}

Guid Transform::getEntityGuid() const
{
    return mEntityGuid;
}

Guid Transform::getGuid() const
{
    return mGuid;
}

Id Transform::getId() const
{
    return mId;
}

void Transform::setPosition(const glm::vec3 &position)
{
    mWorld->getActiveScene()->setTransformPosition(mId, position);
}

void Transform::setRotation(const glm::quat &rotation)
{
    mWorld->getActiveScene()->setTransformRotation(mId, rotation);
}

void Transform::setScale(const glm::vec3 &scale)
{
    assert(scale.x >= 0.001f);
    assert(scale.y >= 0.001f);
    assert(scale.z >= 0.001f);
    mWorld->getActiveScene()->setTransformScale(mId, scale);
}

glm::vec3 Transform::getPosition() const
{
    return mWorld->getActiveScene()->getTransformPosition(mId);
}

glm::quat Transform::getRotation() const
{
    return mWorld->getActiveScene()->getTransformRotation(mId);
}

glm::vec3 Transform::getScale() const
{
    return mWorld->getActiveScene()->getTransformScale(mId);
}

glm::mat4 Transform::getModelMatrix() const
{
    return mWorld->getActiveScene()->getTransformModelMatrix(mId);
}

glm::vec3 Transform::getForward() const
{
    // a transform with zero rotation has its blue axis pointing in the +z direction
    return mWorld->getActiveScene()->getTransformForward(mId);
}

glm::vec3 Transform::getUp() const
{
    // a transform with zero rotation has its green axis pointing in the +y direction
    return mWorld->getActiveScene()->getTransformUp(mId);
}

glm::vec3 Transform::getRight() const
{
    // a transform with zero rotation has its red axis pointing in the +x direction
    return mWorld->getActiveScene()->getTransformRight(mId);
}

bool Transform::decompose(const glm::mat4 &model, glm::vec3 &translation, glm::quat &rotation, glm::vec3 &scale)
{
    glm::vec3 skew;
    glm::vec4 perspective;
    return glm::decompose(model, scale, rotation, translation, skew, perspective);
}

void Transform::v3Scale(glm::vec3 &v, float desiredLength)
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

Entity *Transform::getEntity() const
{
    return mWorld->getActiveScene()->getEntityByGuid(mEntityGuid);
}