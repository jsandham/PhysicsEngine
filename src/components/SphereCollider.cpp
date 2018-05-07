#include "SphereCollider.h"

#include "../core/Geometry.h"

using namespace PhysicsEngine;

SphereCollider::SphereCollider()
{

}

SphereCollider::SphereCollider(Entity *entity)
{
	this->entity = entity;
}

SphereCollider::~SphereCollider()
{

}

bool SphereCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(this->sphere, bounds);
}