#include "../../include/components/CapsuleCollider.h"

#include "../../include/core/Intersect.h"
#include "../../include/core/Serialization.h"

using namespace PhysicsEngine;

CapsuleCollider::CapsuleCollider(World* world) : Collider(world)
{
}

CapsuleCollider::CapsuleCollider(World* world, const Guid& id) : Collider(world, id)
{
}

CapsuleCollider::~CapsuleCollider()
{
}

void CapsuleCollider::serialize(YAML::Node &out) const
{
    Collider::serialize(out);

    out["capsule"] = mCapsule;
}

void CapsuleCollider::deserialize(const YAML::Node &in)
{
    Collider::deserialize(in);

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

bool CapsuleCollider::intersect(AABB aabb) const
{
    return Intersect::intersect(aabb, mCapsule);
}