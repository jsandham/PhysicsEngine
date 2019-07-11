#include <math.h> 

#include "../../include/components/SphereCollider.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

SphereCollider::SphereCollider()
{

}

SphereCollider::SphereCollider(std::vector<char> data)
{
	deserialize(data);
}

SphereCollider::~SphereCollider()
{

}

std::vector<char> SphereCollider::serialize()
{
	SphereColliderHeader header;
	header.componentId = componentId;
	header.entityId = entityId;
	header.sphere = sphere;

	int numberOfBytes = sizeof(SphereColliderHeader);

	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &header, sizeof(SphereColliderHeader));

	return data;
}

void SphereCollider::deserialize(std::vector<char> data)
{
	SphereColliderHeader* header = reinterpret_cast<SphereColliderHeader*>(&data[0]);

	componentId = header->componentId;
	entityId = header->entityId;
	sphere = header->sphere;
}

bool SphereCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(this->sphere, bounds);
}

std::vector<float> SphereCollider::getLines() const
{
	std::vector<float> lines;

	float pi = 3.14159265f;

	for(int i = 0; i < 36; i++){
		float theta1 = i * 10.0f;
		float theta2 = (i + 1) * 10.0f;

		lines.push_back(sphere.centre.x + sphere.radius * cos((pi / 180.0f) * theta1));
		lines.push_back(sphere.centre.y + sphere.radius * sin((pi / 180.0f) * theta1));
		lines.push_back(sphere.centre.z);
		lines.push_back(sphere.centre.x + sphere.radius * cos((pi / 180.0f) * theta2));
		lines.push_back(sphere.centre.y + sphere.radius * sin((pi / 180.0f) * theta2));
		lines.push_back(sphere.centre.z);
	}

	for(int i = 0; i < 36; i++){
		float theta1 = i * 10.0f;
		float theta2 = (i + 1) * 10.0f;

		lines.push_back(sphere.centre.x);
		lines.push_back(sphere.centre.y + sphere.radius * sin((pi / 180.0f) * theta1));
		lines.push_back(sphere.centre.z + sphere.radius * cos((pi / 180.0f) * theta1));
		lines.push_back(sphere.centre.x);
		lines.push_back(sphere.centre.y + sphere.radius * sin((pi / 180.0f) * theta2));
		lines.push_back(sphere.centre.z + sphere.radius * cos((pi / 180.0f) * theta2));
	}

	for(int i = 0; i < 36; i++){
		float theta1 = i * 10.0f;
		float theta2 = (i + 1) * 10.0f;

		lines.push_back(sphere.centre.x + sphere.radius * cos((pi / 180.0f) * theta1));
		lines.push_back(sphere.centre.y);
		lines.push_back(sphere.centre.z + sphere.radius * sin((pi / 180.0f) * theta1));
		lines.push_back(sphere.centre.x + sphere.radius * cos((pi / 180.0f) * theta2));
		lines.push_back(sphere.centre.y);
		lines.push_back(sphere.centre.z + sphere.radius * sin((pi / 180.0f) * theta2));
	}

	return lines;
}