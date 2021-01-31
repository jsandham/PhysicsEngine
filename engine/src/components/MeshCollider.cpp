#include "../../include/components/MeshCollider.h"

#include "../../include/core/Intersect.h"
#include "../../include/core/Serialize.h"

using namespace PhysicsEngine;

MeshCollider::MeshCollider() : Collider()
{
    mMeshId = Guid::INVALID;
}

MeshCollider::MeshCollider(Guid id) : Collider(id)
{
    mMeshId = Guid::INVALID;
}

MeshCollider::~MeshCollider()
{
}

std::vector<char> MeshCollider::serialize() const
{
    return serialize(mId, mEntityId);
}

std::vector<char> MeshCollider::serialize(const Guid &componentId, const Guid &entityId) const
{
    MeshColliderHeader header;
    header.mComponentId = componentId;
    header.mEntityId = entityId;
    header.mMeshId = mMeshId;

    std::vector<char> data(sizeof(MeshColliderHeader));

    memcpy(&data[0], &header, sizeof(MeshColliderHeader));

    return data;
}

void MeshCollider::deserialize(const std::vector<char> &data)
{
    const MeshColliderHeader *header = reinterpret_cast<const MeshColliderHeader *>(&data[0]);

    mId = header->mComponentId;
    mEntityId = header->mEntityId;
    mMeshId = header->mMeshId;
}

void MeshCollider::serialize(std::ostream& out) const
{
    Collider::serialize(out);

    PhysicsEngine::write<Guid>(out, mMeshId);
}

void MeshCollider::deserialize(std::istream& in)
{
    Collider::deserialize(in);

    PhysicsEngine::read(in, mMeshId);
}

bool MeshCollider::intersect(AABB aabb) const
{
    return false;
}