#include "../../include/components/MeshCollider.h"

#include "../../include/core/Intersect.h"

using namespace PhysicsEngine;

MeshCollider::MeshCollider(World* world) : Collider(world)
{
    mMeshId = Guid::INVALID;
}

MeshCollider::MeshCollider(World* world, const Guid& id) : Collider(world, id)
{
    mMeshId = Guid::INVALID;
}

MeshCollider::~MeshCollider()
{
}

void MeshCollider::serialize(YAML::Node &out) const
{
    Collider::serialize(out);

    out["meshId"] = mMeshId;
}

void MeshCollider::deserialize(const YAML::Node &in)
{
    Collider::deserialize(in);

    mMeshId = YAML::getValue<Guid>(in, "meshId");
}

int MeshCollider::getType() const
{
    return PhysicsEngine::MESHCOLLIDER_TYPE;
}

std::string MeshCollider::getObjectName() const
{
    return PhysicsEngine::MESHCOLLIDER_NAME;
}

bool MeshCollider::intersect(AABB aabb) const
{
    return false;
}