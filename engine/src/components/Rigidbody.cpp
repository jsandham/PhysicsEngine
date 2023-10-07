#include "../../include/components/Rigidbody.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/World.h"
#include "../../include/core/GLM.h"

using namespace PhysicsEngine;

RigidbodyData::RigidbodyData()
{
    mVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
    mCentreOfMass = glm::vec3(0.0f, 0.0f, 0.0f);
    mAngularVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
    mInertiaTensor = glm::mat3(1.0f);

    mHalfVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

    mUseGravity = true;
    mMass = 1.0f;
    mDrag = 0.0f;
    mAngularDrag = 0.05f;
}

void RigidbodyData::serialize(YAML::Node &out) const
{
    out["useGravity"] = mUseGravity;
    out["mass"] = mMass;
    out["drag"] = mDrag;
    out["angularDrag"] = mAngularDrag;
    out["velocity"] = mVelocity;
    out["angularVelocity"] = mAngularVelocity;
    out["centreOfMass"] = mCentreOfMass;
}

void RigidbodyData::deserialize(const YAML::Node &in)
{
    mUseGravity = YAML::getValue<bool>(in, "useGravity");
    mMass = YAML::getValue<float>(in, "mass");
    mDrag = YAML::getValue<float>(in, "drag");
    mAngularDrag = YAML::getValue<float>(in, "angularDrag");
    mVelocity = YAML::getValue<glm::vec3>(in, "velocity");
    mAngularVelocity = YAML::getValue<glm::vec3>(in, "angularVelocity");
    mCentreOfMass = YAML::getValue<glm::vec3>(in, "centreOfMass");
}

Rigidbody::Rigidbody(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mEnabled = true;
}

Rigidbody::Rigidbody(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mEnabled = true;
}

Rigidbody::~Rigidbody()
{
}

void Rigidbody::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["entityId"] = mEntityGuid;

    out["enabled"] = mEnabled;
}

void Rigidbody::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

    mEntityGuid = YAML::getValue<Guid>(in, "entityId");

    mEnabled = YAML::getValue<bool>(in, "enabled");
}

int Rigidbody::getType() const
{
    return PhysicsEngine::RIGIDBODY_TYPE;
}

std::string Rigidbody::getObjectName() const
{
    return PhysicsEngine::RIGIDBODY_NAME;
}

Guid Rigidbody::getEntityGuid() const
{
    return mEntityGuid;
}

Guid Rigidbody::getGuid() const
{
    return mGuid;
}

Id Rigidbody::getId() const
{
    return mId;
}