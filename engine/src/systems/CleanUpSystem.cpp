#include "../../include/systems/CleanUpSystem.h"

#include "../../include/core/Log.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

CleanUpSystem::CleanUpSystem(World* world) : System(world)
{
}

CleanUpSystem::CleanUpSystem(World* world, const Guid& id) : System(world, id)
{
}

CleanUpSystem::~CleanUpSystem()
{
}

void CleanUpSystem::serialize(YAML::Node &out) const
{
    System::serialize(out);
}

void CleanUpSystem::deserialize(const YAML::Node &in)
{
    System::deserialize(in);
}

int CleanUpSystem::getType() const
{
    return PhysicsEngine::CLEANUPSYSTEM_TYPE;
}

std::string CleanUpSystem::getObjectName() const
{
    return PhysicsEngine::CLEANUPSYSTEM_NAME;
}

void CleanUpSystem::init(World *world)
{
    mWorld = world;
}

void CleanUpSystem::update(const Input &input, const Time &time)
{
    std::vector<std::tuple<Guid, Guid, int>> componentIdsMarkedLatentDestroy =
        mWorld->getActiveScene()->getComponentIdsMarkedLatentDestroy();
    for (size_t i = 0; i < componentIdsMarkedLatentDestroy.size(); i++)
    {

        mWorld->getActiveScene()->immediateDestroyComponent(std::get<0>(componentIdsMarkedLatentDestroy[i]),
                                          std::get<1>(componentIdsMarkedLatentDestroy[i]),
                                          std::get<2>(componentIdsMarkedLatentDestroy[i]));
    }

    std::vector<Guid> entityIdsMarkedForLatentDestroy = mWorld->getActiveScene()->getEntityIdsMarkedLatentDestroy();
    for (size_t i = 0; i < entityIdsMarkedForLatentDestroy.size(); i++)
    {
        mWorld->getActiveScene()->immediateDestroyEntity(entityIdsMarkedForLatentDestroy[i]);
    }

    mWorld->getActiveScene()->clearIdsMarkedCreatedOrDestroyed();
}