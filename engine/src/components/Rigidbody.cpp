#include "../../include/components/Rigidbody.h"

#include "../../include/core/Serialization.h"

using namespace PhysicsEngine;

Rigidbody::Rigidbody(World* world) : Component(world)
{
    mUseGravity = true;
    mEnabled = true;
    mMass = 1.0f;
    mDrag = 0.0f;
    mAngularDrag = 0.05f;

    mVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
    mCentreOfMass = glm::vec3(0.0f, 0.0f, 0.0f);
    mAngularVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
    mInertiaTensor = glm::mat3(1.0f);

    mHalfVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
}

Rigidbody::Rigidbody(World* world, const Guid& id) : Component(world, id)
{
    mUseGravity = true;
    mEnabled = true;
    mMass = 1.0f;
    mDrag = 0.0f;
    mAngularDrag = 0.05f;

    mVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
    mCentreOfMass = glm::vec3(0.0f, 0.0f, 0.0f);
    mAngularVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
    mInertiaTensor = glm::mat3(1.0f);

    mHalfVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
}

Rigidbody::~Rigidbody()
{
}

void Rigidbody::serialize(YAML::Node &out) const
{
    Component::serialize(out);

    out["useGravity"] = mUseGravity;
    out["enabled"] = mEnabled;
    out["mass"] = mMass;
    out["drag"] = mDrag;
    out["angularDrag"] = mAngularDrag;
    out["velocity"] = mVelocity;
    out["angularVelocity"] = mAngularVelocity;
    out["centreOfMass"] = mCentreOfMass;
}

void Rigidbody::deserialize(const YAML::Node &in)
{
    Component::deserialize(in);

    mUseGravity = YAML::getValue<bool>(in, "useGravity");
    mEnabled = YAML::getValue<bool>(in, "enabled");
    mMass = YAML::getValue<float>(in, "mass");
    mDrag = YAML::getValue<float>(in, "drag");
    mAngularDrag = YAML::getValue<float>(in, "angularDrag");
    mVelocity = YAML::getValue<glm::vec3>(in, "velocity");
    mAngularVelocity = YAML::getValue<glm::vec3>(in, "angularVelocity");
    mCentreOfMass = YAML::getValue<glm::vec3>(in, "centreOfMass");
}

int Rigidbody::getType() const
{
    return PhysicsEngine::RIGIDBODY_TYPE;
}

std::string Rigidbody::getObjectName() const
{
    return PhysicsEngine::RIGIDBODY_NAME;
}