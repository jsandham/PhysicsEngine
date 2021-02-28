#include <iostream>

#include "../../include/components/Collider.h"

using namespace PhysicsEngine;

Collider::Collider() : Component()
{
}

Collider::Collider(Guid id) : Component(id)
{
}

Collider::~Collider()
{
}

void Collider::serialize(std::ostream &out) const
{
    Component::serialize(out);
}

void Collider::deserialize(std::istream &in)
{
    Component::deserialize(in);
}

void Collider::serialize(YAML::Node& out) const
{
    Component::serialize(out);
}

void Collider::deserialize(const YAML::Node& in)
{
    Component::deserialize(in);
}