#include "../../include/components/MeshCollider.h"

#include "../../include/core/Intersect.h"
#include "../../include/core/Serialization.h"

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

void MeshCollider::serialize(std::ostream &out) const
{
    Collider::serialize(out);

    PhysicsEngine::write<Guid>(out, mMeshId);
}

void MeshCollider::deserialize(std::istream &in)
{
    Collider::deserialize(in);

    PhysicsEngine::read(in, mMeshId);
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