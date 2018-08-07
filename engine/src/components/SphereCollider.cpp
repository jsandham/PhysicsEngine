#include "../../include/components/SphereCollider.h"

#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

SphereCollider::SphereCollider()
{

}

SphereCollider::~SphereCollider()
{

}

bool SphereCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(this->sphere, bounds);
}