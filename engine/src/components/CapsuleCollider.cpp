#include "../../include/components/CapsuleCollider.h"

#include "../../include/core/Intersect.h"
#include "../../include/core/Serialize.h"

using namespace PhysicsEngine;

CapsuleCollider::CapsuleCollider() : Collider()
{
}

CapsuleCollider::CapsuleCollider(Guid id) : Collider(id)
{
}

CapsuleCollider::~CapsuleCollider()
{
}

std::vector<char> CapsuleCollider::serialize() const
{
    return serialize(mId, mEntityId);
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

    mId = header->mComponentId;
    mEntityId = header->mEntityId;
    mCapsule = header->mCapsule;
}

void CapsuleCollider::serialize(std::ostream& out) const
{
    Collider::serialize(out);

    PhysicsEngine::write<Capsule>(out, mCapsule);
}

void CapsuleCollider::deserialize(std::istream& in)
{
    Collider::deserialize(in);

    PhysicsEngine::read(in, mCapsule);
}

bool CapsuleCollider::intersect(AABB aabb) const
{
    return Intersect::intersect(aabb, mCapsule);
}