#include "../../include/systems/CleanUpSystem.h"

#include "../../include/core/Log.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Serialization.h"
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

void CleanUpSystem::serialize(std::ostream &out) const
{
    System::serialize(out);
}

void CleanUpSystem::deserialize(std::istream &in)
{
    System::deserialize(in);
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
        mWorld->getComponentIdsMarkedLatentDestroy();
    for (size_t i = 0; i < componentIdsMarkedLatentDestroy.size(); i++)
    {

        mWorld->immediateDestroyComponent(std::get<0>(componentIdsMarkedLatentDestroy[i]),
                                          std::get<1>(componentIdsMarkedLatentDestroy[i]),
                                          std::get<2>(componentIdsMarkedLatentDestroy[i]));
    }

    std::vector<Guid> entityIdsMarkedForLatentDestroy = mWorld->getEntityIdsMarkedLatentDestroy();
    for (size_t i = 0; i < entityIdsMarkedForLatentDestroy.size(); i++)
    {
        mWorld->immediateDestroyEntity(entityIdsMarkedForLatentDestroy[i]);
    }

    mWorld->clearIdsMarkedCreatedOrDestroyed();
}