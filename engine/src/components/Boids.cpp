#include <iostream>

#include "../../include/components/Boids.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

Boids::Boids()
{
	meshId = Guid::INVALID;
	materialId = Guid::INVALID;
	numBoids = 0;
	h = 1.0f;
	bounds.centre = glm::vec3(0.0f, 0.0f, 0.0f);
	bounds.size = glm::vec3(1.0f, 1.0f, 1.0f);

	modelMatrices = new glm::mat4[numBoids];
}

Boids::Boids(std::vector<char> data)
{
	size_t index = sizeof(int);
	index += sizeof(char);
	BoidsHeader* header = reinterpret_cast<BoidsHeader*>(&data[index]);

	componentId = header->componentId;
	entityId = header->entityId;
	meshId = header->meshId;
	materialId = header->materialId;
	numBoids = header->numBoids;
	h = header->h;
	bounds = header->bounds;

	modelMatrices = new glm::mat4[numBoids];
}

Boids::~Boids()
{
	delete [] modelMatrices;
}