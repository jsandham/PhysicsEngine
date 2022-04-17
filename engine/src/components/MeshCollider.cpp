#include "../../include/components/MeshCollider.h"

#include "../../include/core/Intersect.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

MeshCollider::MeshCollider(World* world) : Collider(world)
{
    mMeshId = -1;
}

MeshCollider::MeshCollider(World* world, Id id) : Collider(world, id)
{
    mMeshId = -1;
}

MeshCollider::~MeshCollider()
{
}

void MeshCollider::serialize(YAML::Node &out) const
{
    Collider::serialize(out);

    out["meshId"] = mWorld->getGuidOf(mMeshId);
}

void MeshCollider::deserialize(const YAML::Node &in)
{
    Collider::deserialize(in);

    mMeshId = mWorld->getIdOf(YAML::getValue<Guid>(in, "meshId"));
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