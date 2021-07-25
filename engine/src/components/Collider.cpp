#include <iostream>

#include "../../include/components/Collider.h"

using namespace PhysicsEngine;

Collider::Collider(World* world) : Component(world)
{
}

Collider::Collider(World* world, Guid id) : Component(world, id)
{
}

Collider::~Collider()
{
}

void Collider::serialize(YAML::Node &out) const
{
    Component::serialize(out);
}

void Collider::deserialize(const YAML::Node &in)
{
    Component::deserialize(in);
}