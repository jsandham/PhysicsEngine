#include "../../include/components/Rigidbody.h"

using namespace PhysicsEngine;

Rigidbody::Rigidbody()
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

Rigidbody::Rigidbody(std::vector<char> data)
{
	deserialize(data);
}

Rigidbody::~Rigidbody()
{

}

std::vector<char> Rigidbody::serialize() const
{
	return serialize(mComponentId, mEntityId);
}

std::vector<char> Rigidbody::serialize(Guid componentId, Guid entityId) const
{
	RigidbodyHeader header;
	header.mComponentId = componentId;
	header.mEntityId = entityId;
	header.mUseGravity = mUseGravity;
	header.mMass = mMass;
	header.mDrag = mDrag;
	header.mAngularDrag = mAngularDrag;

	header.mVelocity = mVelocity;
	header.mAngularVelocity = mAngularVelocity;
	header.mCentreOfMass = mCentreOfMass;
	header.mInertiaTensor = mInertiaTensor;

	header.mHalfVelocity = mHalfVelocity;

	std::vector<char> data(sizeof(RigidbodyHeader));

	memcpy(&data[0], &header, sizeof(RigidbodyHeader));

	return data;
}

void Rigidbody::deserialize(std::vector<char> data)
{
	RigidbodyHeader* header = reinterpret_cast<RigidbodyHeader*>(&data[0]);

	mComponentId = header->mComponentId;
	mEntityId = header->mEntityId;
	mUseGravity = header->mUseGravity;
	mMass = header->mMass;
	mDrag = header->mDrag;
	mAngularDrag = header->mAngularDrag;
	mVelocity = header->mVelocity;
	mAngularVelocity = header->mAngularVelocity;
	mCentreOfMass = header->mCentreOfMass;
	mInertiaTensor = header->mInertiaTensor;

	mHalfVelocity = header->mHalfVelocity;
}