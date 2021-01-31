#include "../../include/components/Rigidbody.h"

#include "../../include/core/Serialize.h"

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

std::vector<char> Rigidbody::serialize() const
{
    return serialize(mId, mEntityId);
}

std::vector<char> Rigidbody::serialize(const Guid &componentId, const Guid &entityId) const
{
    RigidbodyHeader header;
    header.mComponentId = componentId;
    header.mEntityId = entityId;
    header.mUseGravity = static_cast<uint8_t>(mUseGravity);
    header.mMass = mMass;
    header.mDrag = mDrag;
    header.mAngularDrag = mAngularDrag;

    header.mVelocity = mVelocity;
    header.mAngularVelocity = mAngularVelocity;
    header.mCentreOfMass = mCentreOfMass;
    header.mInertiaTensor = mInertiaTensor;

    std::vector<char> data(sizeof(RigidbodyHeader));

    memcpy(&data[0], &header, sizeof(RigidbodyHeader));

    return data;
}

void Rigidbody::deserialize(const std::vector<char> &data)
{
    const RigidbodyHeader *header = reinterpret_cast<const RigidbodyHeader *>(&data[0]);

    mId = header->mComponentId;
    mEntityId = header->mEntityId;
    mUseGravity = static_cast<bool>(header->mUseGravity);
    mMass = header->mMass;
    mDrag = header->mDrag;
    mAngularDrag = header->mAngularDrag;
    mVelocity = header->mVelocity;
    mAngularVelocity = header->mAngularVelocity;
    mCentreOfMass = header->mCentreOfMass;
    mInertiaTensor = header->mInertiaTensor;
}

void Rigidbody::serialize(std::ostream& out) const
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

void Rigidbody::deserialize(std::istream& in)
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

