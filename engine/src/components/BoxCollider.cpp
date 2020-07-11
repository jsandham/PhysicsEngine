#include "../../include/components/BoxCollider.h"

#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

BoxCollider::BoxCollider()
{

}

BoxCollider::BoxCollider(const std::vector<char>& data)
{
	deserialize(data);
}

BoxCollider::~BoxCollider()
{

}

std::vector<char> BoxCollider::serialize() const
{
	return serialize(mComponentId, mEntityId);
}

std::vector<char> BoxCollider::serialize(Guid componentId, Guid entityId) const
{
	BoxColliderHeader header;
	header.mComponentId = componentId;
	header.mEntityId = entityId;
	header.mAABB = mAABB;

	std::vector<char> data(sizeof(BoxColliderHeader));

	memcpy(&data[0], &header, sizeof(BoxColliderHeader));

	return data;
}

void BoxCollider::deserialize(const std::vector<char>& data)
{
	const BoxColliderHeader* header = reinterpret_cast<const BoxColliderHeader*>(&data[0]);

	mComponentId = header->mComponentId;
	mEntityId = header->mEntityId;
	mAABB = header->mAABB;
}

bool BoxCollider::intersect(AABB aabb) const
{
	return Geometry::intersect(mAABB, aabb);
}

std::vector<float> BoxCollider::getLines() const  //might not want to store lines in class so if I end up doing that, instead move this to a utility method??
{
	std::vector<float> lines;
	glm::vec3 centre = mAABB.mCentre;
	glm::vec3 extents = mAABB.getExtents();

	float xf[] = {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f};
	float yf[] = {1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f};
	float zf[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

	for(int i = 0; i < 8; i++){
		lines.push_back(centre.x + xf[i] * extents.x);
		lines.push_back(centre.y + yf[i] * extents.y);
		lines.push_back(centre.z + zf[i] * extents.z);
	}

	for(int i = 0; i < 8; i++){
		lines.push_back(centre.x + xf[i] * extents.x);
		lines.push_back(centre.y + yf[i] * extents.y);
		lines.push_back(centre.z - zf[i] * extents.z);
	}

	float xg[] = {-1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f};
	float yg[] = {1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
	float zg[] = {-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f};

	for(int i = 0; i < 8; i++){
		lines.push_back(centre.x + xg[i] * extents.x);
		lines.push_back(centre.y + yg[i] * extents.y);
		lines.push_back(centre.z + zg[i] * extents.z);
	}


	// lines.push_back(centre.x - extents.x);
	// lines.push_back(centre.y - extents.y);
	// lines.push_back(centre.z + extents.z);

	// lines.push_back(centre.x + extents.x);
	// lines.push_back(centre.y - extents.y);
	// lines.push_back(centre.z + extents.z);

	// lines.push_back(centre.x + extents.x);
	// lines.push_back(centre.y - extents.y);
	// lines.push_back(centre.z + extents.z);

	// lines.push_back(centre.x + extents.x);
	// lines.push_back(centre.y + extents.y);
	// lines.push_back(centre.z + extents.z);

	// lines.push_back(centre.x + extents.x);
	// lines.push_back(centre.y + extents.y);
	// lines.push_back(centre.z + extents.z);

	// lines.push_back(centre.x - extents.x);
	// lines.push_back(centre.y + extents.y);
	// lines.push_back(centre.z + extents.z);

	// lines.push_back(centre.x - extents.x);
	// lines.push_back(centre.y + extents.y);
	// lines.push_back(centre.z + extents.z);

	// lines.push_back(centre.x - extents.x);
	// lines.push_back(centre.y - extents.y);
	// lines.push_back(centre.z + extents.z);





	return lines;
}