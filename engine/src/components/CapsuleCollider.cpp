#include "../../include/components/CapsuleCollider.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

CapsuleCollider::CapsuleCollider()
{
	
}

CapsuleCollider::CapsuleCollider(std::vector<char> data)
{
	size_t index = sizeof(char);
	index += sizeof(int);
	CapsuleColliderHeader* header = reinterpret_cast<CapsuleColliderHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	capsule = header->capsule;
}

CapsuleCollider::~CapsuleCollider()
{

}

bool CapsuleCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(bounds, this->capsule);
}