#include "../../include/components/SphereCollider.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

SphereCollider::SphereCollider()
{

}

SphereCollider::SphereCollider(std::vector<char> data)
{

}

SphereCollider::~SphereCollider()
{

}

void* SphereCollider::operator new(size_t size)
{
	return getAllocator<SphereCollider>().allocate();
}

void SphereCollider::operator delete(void*)
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