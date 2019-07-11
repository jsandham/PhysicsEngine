#include "../../include/components/MeshCollider.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

MeshCollider::MeshCollider()
{

}

MeshCollider::MeshCollider(std::vector<char> data)
{
	deserialize(data);
}

MeshCollider::~MeshCollider()
{

}

std::vector<char> MeshCollider::serialize()
{
	MeshColliderHeader header;
	header.componentId = componentId;
	header.entityId = entityId;
	header.meshId = meshId;

	int numberOfBytes = sizeof(MeshColliderHeader);

	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &header, sizeof(MeshColliderHeader));

	return data;
}

void MeshCollider::deserialize(std::vector<char> data)
{
	MeshColliderHeader* header = reinterpret_cast<MeshColliderHeader*>(&data[0]);

	componentId = header->componentId;
	entityId = header->entityId;
	meshId = header->meshId;
}

bool MeshCollider::intersect(Bounds bounds)
{
	return false;
}