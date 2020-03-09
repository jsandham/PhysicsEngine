#include "../../include/components/MeshCollider.h"

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

std::vector<char> MeshCollider::serialize() const
{
	return serialize(componentId, entityId);
}

std::vector<char> MeshCollider::serialize(Guid componentId, Guid entityId) const
{
	MeshColliderHeader header;
	header.componentId = componentId;
	header.entityId = entityId;
	header.meshId = meshId;

	std::vector<char> data(sizeof(MeshColliderHeader));

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

bool MeshCollider::intersect(Bounds bounds) const
{
	return false;
}