#include "../../include/core/UniformGrid.h"

using namespace PhysicsEngine;

UniformGrid::UniformGrid()
{

}

UniformGrid::~UniformGrid()
{

}

void UniformGrid::create(glm::vec3 gridSize, int hashTableSize)
{
	this->gridSize = gridSize;
	this->hashTableSize = hashTableSize;

	hashTable.resize(hashTableSize, 0);

	for(size_t i = 0; i < ){

	}
}

void UniformGrid::firstPass(std::vector<Sphere> spheres)
{
	for(size_t i = 0; i < sphere.size(); i++){
		int hash = computeHash(spheres[i].centre);

		hashTable[hash]++;
	}

	int totalCount = 0;
	for(size_t i = 0; i < hashTable.size(); i++){
		totalCount += hashTable[i];
	}

	objects.resize(totalCount);
	for(size_t i = 0; i < objects.size(); i++){
		objects[i].sphere.centre = glm::vec3(0.0f, 0.0f, 0.0f);
		objects[i].sphere.radius = 0.0f;
		objects[i].id = Guid::INVALID;
	}

	totalCount = 0;
	for(size_t i = 0; i < hashTable.size(); i++){
		if(hashTable[i] > 0){
			int count = hashTable[i];
			hashTable[i] = totalCount;
			totalCount += count;
		}
	}
}

void UniformGrid::secondPass(std::vector<Sphere>, std::vector<Guid> ids)
{

}

Object* UniformGrid::intersect(Ray ray)
{
	return NULL;
}

// int UniformGrid::computeHash(glm::vec3 centre)
int UniformGrid::computeHash(glm::ivec3 centre)
{		
	const int h1 = 0x8da6b343;
	const int h2 = 0xd8163841;
	const int h3 = 0xcb1ab31f;

	int n = h1 * centre.x + h2 * centre.y + h3 * centre.z;
	n = n % hashTableSize;
	if(n < 0){
		n += hashTableSize;
	}

	return n;
}