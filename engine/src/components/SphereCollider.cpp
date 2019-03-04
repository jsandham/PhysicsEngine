#include "../../include/components/SphereCollider.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

SphereCollider::SphereCollider()
{

}

SphereCollider::SphereCollider(std::vector<char> data)
{
	size_t index = sizeof(int);
	index += sizeof(char);
	SphereColliderHeader* header = reinterpret_cast<SphereColliderHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	sphere = header->sphere;
}

SphereCollider::~SphereCollider()
{

}

bool SphereCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(this->sphere, bounds);
}