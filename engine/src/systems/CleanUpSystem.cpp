#include "../../include/systems/CleanUpSystem.h"

#include "../../include/core/Log.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

CleanUpSystem::CleanUpSystem() : System()
{
}

CleanUpSystem::CleanUpSystem(Guid id) : System(id)
{
}

CleanUpSystem::~CleanUpSystem()
{
}

std::vector<char> CleanUpSystem::serialize() const
{
    return serialize(mId);
}

std::vector<char> CleanUpSystem::serialize(const Guid &systemId) const
{
    CleanUpSystemHeader header;
    header.mSystemId = systemId;
    header.mUpdateOrder = static_cast<int32_t>(mOrder);

    std::vector<char> data(sizeof(CleanUpSystemHeader));

    memcpy(&data[0], &header, sizeof(CleanUpSystemHeader));

    return data;
}

void CleanUpSystem::deserialize(const std::vector<char> &data)
{
    const CleanUpSystemHeader *header = reinterpret_cast<const CleanUpSystemHeader *>(&data[0]);

    mId = header->mSystemId;
    mOrder = static_cast<int>(header->mUpdateOrder);
}

void CleanUpSystem::init(World *world)
{
    mWorld = world;
}

void CleanUpSystem::update(const Input &input, const Time &time)
{
    std::vector<triple<Guid, Guid, int>> componentIdsMarkedLatentDestroy = mWorld->getComponentIdsMarkedLatentDestroy();
    for (size_t i = 0; i < componentIdsMarkedLatentDestroy.size(); i++)
    {

        mWorld->immediateDestroyComponent(componentIdsMarkedLatentDestroy[i].first,
                                          componentIdsMarkedLatentDestroy[i].second,
                                          componentIdsMarkedLatentDestroy[i].third);
    }

    std::vector<Guid> entityIdsMarkedForLatentDestroy = mWorld->getEntityIdsMarkedLatentDestroy();
    for (size_t i = 0; i < entityIdsMarkedForLatentDestroy.size(); i++)
    {
        mWorld->immediateDestroyEntity(entityIdsMarkedForLatentDestroy[i]);
    }

    mWorld->clearIdsMarkedCreatedOrDestroyed();
}