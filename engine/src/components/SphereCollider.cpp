#include "../../include/components/SphereCollider.h"

#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

SphereCollider::SphereCollider()
{

}

SphereCollider::SphereCollider(unsigned char* data)
{

}

SphereCollider::~SphereCollider()
{

}

void SphereCollider::load(SphereColliderData data)
{
	entityId = data.entityId;
	componentId = data.componentId;

	sphere = data.sphere;
}

bool SphereCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(this->sphere, bounds);
}