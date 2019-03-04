#include "../../include/systems/CleanUpSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

CleanUpSystem::CleanUpSystem(std::vector<char> data)
{
	type = 2;
}

CleanUpSystem::~CleanUpSystem()
{

}

void CleanUpSystem::init(World* world)
{
	this->world = world;
}

void CleanUpSystem::update(Input input)
{
	// std::vector<Guid> entityIdsMarkedForLatentDestroy = world->entityIdsMarkedForLatentDestroy;
	// for(int i = 0; i < entityIdsMarkedForLatentDestroy.size(); i++){
	// 	std::cout << "Clean up system attempting to destroy entity " << entityIdsMarkedForLatentDestroy[i].toString() << std::endl;
	// 	world->immediateDestroy(entityIdsMarkedForLatentDestroy[i]):
	// }
}