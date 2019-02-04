#include "../../include/components/BoxCollider.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

BoxCollider::BoxCollider()
{

}

BoxCollider::BoxCollider(unsigned char* data)
{

}

BoxCollider::~BoxCollider()
{

}

void* BoxCollider::operator new(size_t size)
{
	return getAllocator<BoxCollider>().allocate();
}

void BoxCollider::operator delete(void*)
{

}

void BoxCollider::load(BoxColliderData data)
{
	entityId = data.entityId;
	componentId = data.componentId;

	bounds = data.bounds;
}

bool BoxCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(this->bounds, bounds);
}