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

void Collider::serialize(std::ostream& out) const
{
	Component::serialize(out);
}

void Collider::deserialize(std::istream& in)
{
	Component::deserialize(in);
}