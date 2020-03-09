#include "../../include/components/CapsuleCollider.h"

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

std::vector<char> CapsuleCollider::serialize() const
{
	return serialize(componentId, entityId);
}

std::vector<char> CapsuleCollider::serialize(Guid componentId, Guid entityId) const
{
	CapsuleColliderHeader header;
	header.componentId = componentId;
	header.entityId = entityId;
	header.capsule = capsule;

	std::vector<char> data(sizeof(CapsuleColliderHeader));

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

bool CapsuleCollider::intersect(Bounds bounds) const
{
	return Geometry::intersect(bounds, this->capsule);
}