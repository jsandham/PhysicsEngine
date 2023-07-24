#include "../../include/components/MeshCollider.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/World.h"
#include "../../include/core/Intersect.h"

using namespace PhysicsEngine;

MeshCollider::MeshCollider(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mEnabled = true;
    mMeshId = Guid::INVALID;
}

MeshCollider::MeshCollider(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mEnabled = true;
    mMeshId = Guid::INVALID;
}

MeshCollider::~MeshCollider()
{
}

void MeshCollider::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["entityId"] = mEntityGuid;

    out["enabled"] = mEnabled;

    out["meshId"] = mMeshId;
}

void MeshCollider::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

    mEntityGuid = YAML::getValue<Guid>(in, "entityId");

    mEnabled = YAML::getValue<bool>(in, "enabled");

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

Guid MeshCollider::getEntityGuid() const
{
    return mEntityGuid;
}

Guid MeshCollider::getGuid() const
{
    return mGuid;
}

Id MeshCollider::getId() const
{
    return mId;
}

bool MeshCollider::intersect(AABB aabb) const
{
    return false;
}