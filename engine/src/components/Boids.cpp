#include "../../include/components/Boids.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

Boids::Boids()
{
	meshId = Guid::INVALID;
	materialId = Guid::INVALID;
	numBoids = 0;
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
	bounds = header->bounds;
}

Boids::~Boids()
{
	
}

void* Boids::operator new(size_t size)
{
	return getAllocator<Boids>().allocate();
}

void Boids::operator delete(void*)
{

}