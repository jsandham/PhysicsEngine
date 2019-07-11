#include <iostream>

#include "../../include/components/Boids.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

Boids::Boids()
{
	meshId = Guid::INVALID;
	// materialId = Guid::INVALID;
	shaderId = Guid::INVALID;
	numBoids = 0;
	h = 1.0f;
	bounds.centre = glm::vec3(0.0f, 0.0f, 0.0f);
	bounds.size = glm::vec3(1.0f, 1.0f, 1.0f);
}

Boids::Boids(std::vector<char> data)
{
	deserialize(data);
}

Boids::~Boids()
{
}

std::vector<char> Boids::serialize()
{
	BoidsHeader header;
	header.componentId = componentId;
	header.entityId = entityId;
	header.meshId = meshId;
	header.shaderId = shaderId;
	header.numBoids = numBoids;
	header.h = h;
	header.bounds = bounds;

	int numberOfBytes = sizeof(BoidsHeader);

	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &header, sizeof(BoidsHeader));

	return data;
}

void Boids::deserialize(std::vector<char> data)
{
	BoidsHeader* header = reinterpret_cast<BoidsHeader*>(&data[0]);

	componentId = header->componentId;
	entityId = header->entityId;
	meshId = header->meshId;
	// materialId = header->materialId;
	shaderId = header->shaderId;
	numBoids = header->numBoids;
	h = header->h;
	bounds = header->bounds;
}