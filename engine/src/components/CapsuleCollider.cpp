#include "../../include/components/CapsuleCollider.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

CapsuleCollider::CapsuleCollider()
{
	
}

CapsuleCollider::CapsuleCollider(std::vector<char> data)
{
	deserialize(data);
}

CapsuleCollider::~CapsuleCollider()
{

}

std::vector<char> CapsuleCollider::serialize()
{
	CapsuleColliderHeader header;
	header.componentId = componentId;
	header.entityId = entityId;
	header.capsule = capsule;

	int numberOfBytes = sizeof(CapsuleColliderHeader);

	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &header, sizeof(CapsuleColliderHeader));

	return data;
}

void CapsuleCollider::deserialize(std::vector<char> data)
{
	CapsuleColliderHeader* header = reinterpret_cast<CapsuleColliderHeader*>(&data[0]);

	componentId = header->componentId;
	entityId = header->entityId;
	capsule = header->capsule;
}

bool CapsuleCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(bounds, this->capsule);
}