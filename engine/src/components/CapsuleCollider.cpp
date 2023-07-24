#include "../../include/components/CapsuleCollider.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/World.h"
#include "../../include/core/Intersect.h"

using namespace PhysicsEngine;

CapsuleCollider::CapsuleCollider(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mEnabled = true;
}

CapsuleCollider::CapsuleCollider(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mEnabled = true;
}

CapsuleCollider::~CapsuleCollider()
{
}

void CapsuleCollider::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["entityId"] = mEntityGuid;

    out["enabled"] = mEnabled;

    out["capsule"] = mCapsule;
}

void CapsuleCollider::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

    mEntityGuid = YAML::getValue<Guid>(in, "entityId");

    mEnabled = YAML::getValue<bool>(in, "enabled");

    mCapsule = YAML::getValue<Capsule>(in, "capsule");
}

int CapsuleCollider::getType() const
{
    return PhysicsEngine::CAPSULECOLLIDER_TYPE;
}

std::string CapsuleCollider::getObjectName() const
{
    return PhysicsEngine::CAPSULECOLLIDER_NAME;
}

Guid CapsuleCollider::getEntityGuid() const
{
    return mEntityGuid;
}

Guid CapsuleCollider::getGuid() const
{
    return mGuid;
}

Id CapsuleCollider::getId() const
{
    return mId;
}

bool CapsuleCollider::intersect(AABB aabb) const
{
    return Intersect::intersect(aabb, mCapsule);
}