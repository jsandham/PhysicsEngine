#include "../../include/components/BoxCollider.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

BoxCollider::BoxCollider()
{

}

BoxCollider::BoxCollider(std::vector<char> data)
{
	size_t index = sizeof(char);
	index += sizeof(int);
	BoxColliderHeader* header = reinterpret_cast<BoxColliderHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	bounds = header->bounds;
}

BoxCollider::~BoxCollider()
{

}

bool BoxCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(this->bounds, bounds);
}

std::vector<float> BoxCollider::getLines() const  //might not want to store lines in class so if I end up doing that, instead move this to a utility method??
{
	std::vector<float> lines;
	glm::vec3 centre = bounds.centre;
	glm::vec3 extents = bounds.getExtents();

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