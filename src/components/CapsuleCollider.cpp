#include "CapsuleCollider.h"

#include "../core/Geometry.h"

using namespace PhysicsEngine;

CapsuleCollider::CapsuleCollider()
{

}

CapsuleCollider::CapsuleCollider(Entity *entity)
{
	this->entity = entity;
}

CapsuleCollider::~CapsuleCollider()
{

}

bool CapsuleCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(bounds, this->capsule);
}