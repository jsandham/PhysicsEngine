#include "../../include/components/BoxCollider.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

BoxCollider::BoxCollider()
{

}

BoxCollider::BoxCollider(std::vector<char> data)
{
	size_t index = sizeof(int);
	index += sizeof(char);
	BoxColliderHeader* header = reinterpret_cast<BoxColliderHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	bounds = header->bounds;
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

bool BoxCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(this->bounds, bounds);
}