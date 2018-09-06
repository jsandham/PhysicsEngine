#include "../../include/components/CapsuleCollider.h"

#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

CapsuleCollider::CapsuleCollider()
{

}

CapsuleCollider::~CapsuleCollider()
{

}

bool CapsuleCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(bounds, this->capsule);
}