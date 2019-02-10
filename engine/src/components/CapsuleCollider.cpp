#include "../../include/components/CapsuleCollider.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

CapsuleCollider::CapsuleCollider()
{
	
}

CapsuleCollider::CapsuleCollider(std::vector<char> data)
{
	size_t index = sizeof(int);
	index += sizeof(char);
	CapsuleColliderHeader* header = reinterpret_cast<CapsuleColliderHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	capsule = header->capsule;
}

CapsuleCollider::~CapsuleCollider()
{

}

void* CapsuleCollider::operator new(size_t size)
{
	return getAllocator<CapsuleCollider>().allocate();
}

void CapsuleCollider::operator delete(void*)
{

}

bool CapsuleCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(bounds, this->capsule);
}