#include "../../include/components/CapsuleCollider.h"

#include "../../include/core/Intersect.h"

using namespace PhysicsEngine;

CapsuleCollider::CapsuleCollider()
{
}

CapsuleCollider::CapsuleCollider(const std::vector<char> &data)
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

std::vector<char> CapsuleCollider::serialize(const Guid &componentId, const Guid &entityId) const
{
    CapsuleColliderHeader header;
    header.mComponentId = componentId;
    header.mEntityId = entityId;
    header.mCapsule = mCapsule;

    std::vector<char> data(sizeof(CapsuleColliderHeader));

    memcpy(&data[0], &header, sizeof(CapsuleColliderHeader));

    return data;
}

void CapsuleCollider::deserialize(const std::vector<char> &data)
{
    const CapsuleColliderHeader *header = reinterpret_cast<const CapsuleColliderHeader *>(&data[0]);

    mComponentId = header->mComponentId;
    mEntityId = header->mEntityId;
    mCapsule = header->mCapsule;
}

bool CapsuleCollider::intersect(AABB aabb) const
{
    return Intersect::intersect(aabb, mCapsule);
}