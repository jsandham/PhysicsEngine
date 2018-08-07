#include "../../include/components/BoxCollider.h"

#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

BoxCollider::BoxCollider()
{

}

BoxCollider::~BoxCollider()
{

}

bool BoxCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(this->bounds, bounds);
}