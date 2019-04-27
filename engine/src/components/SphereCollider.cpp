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
	size_t index = sizeof(char);
	index += sizeof(int);
	SphereColliderHeader* header = reinterpret_cast<SphereColliderHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	sphere = header->sphere;
}

SphereCollider::~SphereCollider()
{

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