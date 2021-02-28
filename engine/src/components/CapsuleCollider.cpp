#include "../../include/components/CapsuleCollider.h"

#include "../../include/core/Intersect.h"
#include "../../include/core/Serialization.h"

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

void CapsuleCollider::serialize(std::ostream &out) const
{
    Collider::serialize(out);

    PhysicsEngine::write<Capsule>(out, mCapsule);
}

void CapsuleCollider::deserialize(std::istream &in)
{
    Collider::deserialize(in);

    PhysicsEngine::read(in, mCapsule);
}

void CapsuleCollider::serialize(YAML::Node& out) const
{
    Collider::serialize(out);

    out["capsule"] = mCapsule;
}

void CapsuleCollider::deserialize(const YAML::Node& in)
{
    Collider::deserialize(in);

    mCapsule = in["capsule"].as<Capsule>();
}

bool CapsuleCollider::intersect(AABB aabb) const
{
    return Intersect::intersect(aabb, mCapsule);
}