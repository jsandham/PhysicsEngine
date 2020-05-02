#include "../../include/systems/CleanUpSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"
#include "../../include/core/Log.h"

using namespace PhysicsEngine;

CleanUpSystem::CleanUpSystem()
{
	
}

CleanUpSystem::CleanUpSystem(std::vector<char> data)
{
	deserialize(data);
}

CleanUpSystem::~CleanUpSystem()
{

}

std::vector<char> CleanUpSystem::serialize() const
{
	return serialize(mSystemId);
}

std::vector<char> CleanUpSystem::serialize(Guid systemId) const
{
	std::vector<char> data(sizeof(int));

	memcpy(&data[0], &mOrder, sizeof(int));

	return data;
}

void CleanUpSystem::deserialize(std::vector<char> data)
{
	mOrder = *reinterpret_cast<int*>(&data[0]);
}

void CleanUpSystem::init(World* world)
{
	mWorld = world;
}

void CleanUpSystem::update(Input input, Time time)
{
	mWorld->clearIdsMarkedMoved();

	std::vector<triple<Guid, Guid, int>> componentIdsMarkedLatentDestroy = mWorld->getComponentIdsMarkedLatentDestroy();
	for(size_t i = 0; i < componentIdsMarkedLatentDestroy.size(); i++){

		mWorld->immediateDestroyComponent(componentIdsMarkedLatentDestroy[i].first, componentIdsMarkedLatentDestroy[i].second, componentIdsMarkedLatentDestroy[i].third);
	}

	std::vector<Guid> entityIdsMarkedForLatentDestroy = mWorld->getEntityIdsMarkedLatentDestroy();
	for(size_t i = 0; i < entityIdsMarkedForLatentDestroy.size(); i++){
		mWorld->immediateDestroyEntity(entityIdsMarkedForLatentDestroy[i]);
	}

	mWorld->clearIdsMarkedCreatedOrDestroyed();
}