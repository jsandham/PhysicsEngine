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
	world->clearIdsMarkedMoved();

	std::vector<triple<Guid, Guid, itype>> componentIdsMarkedLatentDestroy = world->getComponentIdsMarkedLatentDestroy();
	for(size_t i = 0; i < componentIdsMarkedLatentDestroy.size(); i++){
		world->immediateDestroyComponent(componentIdsMarkedLatentDestroy[i].first, componentIdsMarkedLatentDestroy[i].second, componentIdsMarkedLatentDestroy[i].third);
	}

	std::vector<Guid> entityIdsMarkedForLatentDestroy = world->getEntityIdsMarkedLatentDestroy();
	for(size_t i = 0; i < entityIdsMarkedForLatentDestroy.size(); i++){
		world->immediateDestroyEntity(entityIdsMarkedForLatentDestroy[i]);
	}

	world->clearIdsMarkedCreatedOrDestroyed();
}