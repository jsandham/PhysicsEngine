#include "../../include/systems/CleanUpSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"
#include "../../include/core/Log.h"

using namespace PhysicsEngine;

CleanUpSystem::CleanUpSystem()
{
	
}

CleanUpSystem::CleanUpSystem(const std::vector<char>& data)
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
	CleanUpSystemHeader header;
	header.mSystemId = systemId;
	header.mUpdateOrder = mOrder;

	std::vector<char> data(sizeof(CleanUpSystemHeader));

	memcpy(&data[0], &header, sizeof(CleanUpSystemHeader));

	return data;
}

void CleanUpSystem::deserialize(const std::vector<char>& data)
{
	const CleanUpSystemHeader* header = reinterpret_cast<const CleanUpSystemHeader*>(&data[0]);

	mSystemId = header->mSystemId;
	mOrder = header->mUpdateOrder;
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