#include <math.h> 

#include "../../include/components/SphereCollider.h"

#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

SphereCollider::SphereCollider()
{

}

SphereCollider::SphereCollider(std::vector<char> data)
{
	deserialize(data);
}

SphereCollider::~SphereCollider()
{

}

std::vector<char> SphereCollider::serialize() const
{
	return serialize(mComponentId, mEntityId);
}

std::vector<char> SphereCollider::serialize(Guid componentId, Guid entityId) const
{
	SphereColliderHeader header;
	header.mComponentId = componentId;
	header.mEntityId = entityId;
	header.mSphere = mSphere;

	std::vector<char> data(sizeof(SphereColliderHeader));

	memcpy(&data[0], &header, sizeof(SphereColliderHeader));

	return data;
}

void SphereCollider::deserialize(std::vector<char> data)
{
	SphereColliderHeader* header = reinterpret_cast<SphereColliderHeader*>(&data[0]);

	mComponentId = header->mComponentId;
	mEntityId = header->mEntityId;
	mSphere = header->mSphere;
}

bool SphereCollider::intersect(Bounds bounds) const
{
	return Geometry::intersect(mSphere, bounds);
}

std::vector<float> SphereCollider::getLines() const
{
	std::vector<float> lines;

	float pi = 3.14159265f;

	for(int i = 0; i < 36; i++){
		float theta1 = i * 10.0f;
		float theta2 = (i + 1) * 10.0f;

		lines.push_back(mSphere.mCentre.x + mSphere.mRadius * cos((pi / 180.0f) * theta1));
		lines.push_back(mSphere.mCentre.y + mSphere.mRadius * sin((pi / 180.0f) * theta1));
		lines.push_back(mSphere.mCentre.z);
		lines.push_back(mSphere.mCentre.x + mSphere.mRadius * cos((pi / 180.0f) * theta2));
		lines.push_back(mSphere.mCentre.y + mSphere.mRadius * sin((pi / 180.0f) * theta2));
		lines.push_back(mSphere.mCentre.z);
	}

	for(int i = 0; i < 36; i++){
		float theta1 = i * 10.0f;
		float theta2 = (i + 1) * 10.0f;

		lines.push_back(mSphere.mCentre.x);
		lines.push_back(mSphere.mCentre.y + mSphere.mRadius * sin((pi / 180.0f) * theta1));
		lines.push_back(mSphere.mCentre.z + mSphere.mRadius * cos((pi / 180.0f) * theta1));
		lines.push_back(mSphere.mCentre.x);
		lines.push_back(mSphere.mCentre.y + mSphere.mRadius * sin((pi / 180.0f) * theta2));
		lines.push_back(mSphere.mCentre.z + mSphere.mRadius * cos((pi / 180.0f) * theta2));
	}

	for(int i = 0; i < 36; i++){
		float theta1 = i * 10.0f;
		float theta2 = (i + 1) * 10.0f;

		lines.push_back(mSphere.mCentre.x + mSphere.mRadius * cos((pi / 180.0f) * theta1));
		lines.push_back(mSphere.mCentre.y);
		lines.push_back(mSphere.mCentre.z + mSphere.mRadius * sin((pi / 180.0f) * theta1));
		lines.push_back(mSphere.mCentre.x + mSphere.mRadius * cos((pi / 180.0f) * theta2));
		lines.push_back(mSphere.mCentre.y);
		lines.push_back(mSphere.mCentre.z + mSphere.mRadius * sin((pi / 180.0f) * theta2));
	}

	return lines;
}