#include "../../include/components/MeshCollider.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

MeshCollider::MeshCollider()
{

}

MeshCollider::MeshCollider(std::vector<char> data)
{
	size_t index = sizeof(char);
	index += sizeof(int);
	MeshColliderHeader* header = reinterpret_cast<MeshColliderHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	meshId = header->meshId;
}

MeshCollider::~MeshCollider()
{

}

bool MeshCollider::intersect(Bounds bounds)
{
	return false;
}