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