#include "../../include/components/Rigidbody.h"

#include "../../include/core/Serialization.h"

using namespace PhysicsEngine;

Rigidbody::Rigidbody() : Component()
{
    mUseGravity = true;
    mMass = 1.0f;
    mDrag = 0.0f;
    mAngularDrag = 0.05f;

    mVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
    mCentreOfMass = glm::vec3(0.0f, 0.0f, 0.0f);
    mAngularVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
    mInertiaTensor = glm::mat3(1.0f);

    mHalfVelocity = glm::vec3(0.0f, 0.0f, 0.0f);
}

Rigidbody::Rigidbody(Guid id) : Component(id)
{
    mUseGravity = true;
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

void Rigidbody::serialize(std::ostream &out) const
{
    Component::serialize(out);

    PhysicsEngine::write<bool>(out, mUseGravity);
    PhysicsEngine::write<float>(out, mMass);
    PhysicsEngine::write<float>(out, mDrag);
    PhysicsEngine::write<float>(out, mAngularDrag);
    PhysicsEngine::write<glm::vec3>(out, mVelocity);
    PhysicsEngine::write<glm::vec3>(out, mAngularVelocity);
    PhysicsEngine::write<glm::vec3>(out, mCentreOfMass);
    PhysicsEngine::write<glm::mat3>(out, mInertiaTensor);
}

void Rigidbody::deserialize(std::istream &in)
{
    Component::deserialize(in);

    PhysicsEngine::read<bool>(in, mUseGravity);
    PhysicsEngine::read<float>(in, mMass);
    PhysicsEngine::read<float>(in, mDrag);
    PhysicsEngine::read<float>(in, mAngularDrag);
    PhysicsEngine::read<glm::vec3>(in, mVelocity);
    PhysicsEngine::read<glm::vec3>(in, mAngularVelocity);
    PhysicsEngine::read<glm::vec3>(in, mCentreOfMass);
    PhysicsEngine::read<glm::mat3>(in, mInertiaTensor);
}

void Rigidbody::serialize(YAML::Node& out) const
{
    Component::serialize(out);

    out["useGravity"] = mUseGravity;
    out["mass"] = mMass;
    out["drag"] = mDrag;
    out["angularDrag"] = mAngularDrag;
    out["velocity"] = mVelocity;
    out["angularVelocity"] = mAngularVelocity;
    out["centreOfMass"] = mCentreOfMass;
}

void Rigidbody::deserialize(const YAML::Node& in)
{
    Component::deserialize(in);

    mUseGravity = in["useGravity"].as<bool>();
    mMass = in["mass"].as<float>();
    mDrag = in["drag"].as<float>();
    mAngularDrag = in["angularDrag"].as<float>();
    mVelocity = in["velocity"].as<glm::vec3>();
    mAngularVelocity = in["angularVelocity"].as<glm::vec3>();
    mCentreOfMass = in["centreOfMass"].as<glm::vec3>();
}

int Rigidbody::getType() const
{
    return PhysicsEngine::RIGIDBODY_TYPE;
}

std::string Rigidbody::getObjectName() const
{
    return PhysicsEngine::RIGIDBODY_NAME;
}