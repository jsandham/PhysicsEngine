#include "../../include/systems/CleanUpSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

CleanUpSystem::CleanUpSystem(std::vector<char> data)
{
	size_t index = sizeof(char);
	type = *reinterpret_cast<int*>(&data[index]);
	index += sizeof(int);
	order = *reinterpret_cast<int*>(&data[index]);

	if(type != 2){
		std::cout << "Error: System type (" << type << ") found in data array is invalid" << std::endl;
	}
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