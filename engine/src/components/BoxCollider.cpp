#include "../../include/components/BoxCollider.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

BoxCollider::BoxCollider()
{

}

BoxCollider::BoxCollider(std::vector<char> data)
{
	size_t index = sizeof(char);
	index += sizeof(int);
	BoxColliderHeader* header = reinterpret_cast<BoxColliderHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	bounds = header->bounds;
}

BoxCollider::~BoxCollider()
{

}

bool BoxCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(this->bounds, bounds);
}