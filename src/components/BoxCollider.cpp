#include "BoxCollider.h"

#include "../core/Geometry.h"

using namespace PhysicsEngine;

BoxCollider::BoxCollider()
{

}

BoxCollider::BoxCollider(Entity *entity)
{
	this->entity = entity;
}

BoxCollider::~BoxCollider()
{

}

bool BoxCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(this->bounds, bounds);
}