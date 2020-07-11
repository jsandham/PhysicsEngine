#include "../../include/components/MeshCollider.h"

#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

MeshCollider::MeshCollider()
{
	mMeshId = Guid::INVALID;
}

MeshCollider::MeshCollider(const std::vector<char>& data)
{
	deserialize(data);
}

MeshCollider::~MeshCollider()
{

}

std::vector<char> MeshCollider::serialize() const
{
	return serialize(mComponentId, mEntityId);
}

std::vector<char> MeshCollider::serialize(Guid componentId, Guid entityId) const
{
	MeshColliderHeader header;
	header.mComponentId = componentId;
	header.mEntityId = entityId;
	header.mMeshId = mMeshId;

	std::vector<char> data(sizeof(MeshColliderHeader));

	memcpy(&data[0], &header, sizeof(MeshColliderHeader));

	return data;
}

void MeshCollider::deserialize(const std::vector<char>& data)
{
	const MeshColliderHeader* header = reinterpret_cast<const MeshColliderHeader*>(&data[0]);

	mComponentId = header->mComponentId;
	mEntityId = header->mEntityId;
	mMeshId = header->mMeshId;
}

bool MeshCollider::intersect(AABB aabb) const
{
	return false;
}