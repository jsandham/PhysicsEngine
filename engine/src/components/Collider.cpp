#include <iostream>

#include "../../include/components/Collider.h"

using namespace PhysicsEngine;

Collider::Collider(World *world, const Id &id) : Component(world, id)
{
    mEnabled = true;
}

Collider::Collider(World *world, const Guid &guid, const Id &id) : Component(world, guid, id)
{
    mEnabled = true;
}

Collider::~Collider()
{
}

void Collider::serialize(YAML::Node &out) const
{
    Component::serialize(out);

    out["enabled"] = mEnabled;
}

void Collider::deserialize(const YAML::Node &in)
{
    Component::deserialize(in);

    mEnabled = YAML::getValue<bool>(in, "enabled");
}