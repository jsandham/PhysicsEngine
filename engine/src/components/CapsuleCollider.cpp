#include "../../include/components/CapsuleCollider.h"

#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

CapsuleCollider::CapsuleCollider()
{
	
}

CapsuleCollider::CapsuleCollider(std::vector<char> data)
{
	deserialize(data);
}

CapsuleCollider::~CapsuleCollider()
{

}

std::vector<char> CapsuleCollider::serialize() const
{
	return serialize(mComponentId, mEntityId);
}

std::vector<char> CapsuleCollider::serialize(Guid componentId, Guid entityId) const
{
	CapsuleColliderHeader header;
	header.mComponentId = componentId;
	header.mEntityId = entityId;
	header.mCapsule = mCapsule;

	std::vector<char> data(sizeof(CapsuleColliderHeader));

	memcpy(&data[0], &header, sizeof(CapsuleColliderHeader));

	return data;
}

void CapsuleCollider::deserialize(std::vector<char> data)
{
	CapsuleColliderHeader* header = reinterpret_cast<CapsuleColliderHeader*>(&data[0]);

	mComponentId = header->mComponentId;
	mEntityId = header->mEntityId;
	mCapsule = header->mCapsule;
}

bool CapsuleCollider::intersect(AABB aabb) const
{
	return Geometry::intersect(aabb, mCapsule);
}