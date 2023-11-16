#include "../../include/systems/CleanUpSystem.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/Log.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

CleanUpSystem::CleanUpSystem(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
}

CleanUpSystem::CleanUpSystem(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
}

CleanUpSystem::~CleanUpSystem()
{
}

void CleanUpSystem::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;
}

void CleanUpSystem::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");
}

int CleanUpSystem::getType() const
{
    return PhysicsEngine::CLEANUPSYSTEM_TYPE;
}

std::string CleanUpSystem::getObjectName() const
{
    return PhysicsEngine::CLEANUPSYSTEM_NAME;
}

Guid CleanUpSystem::getGuid() const
{
    return mGuid;
}

Id CleanUpSystem::getId() const
{
    return mId;
}

void CleanUpSystem::init(World *world)
{
    mWorld = world;
}

void CleanUpSystem::update()
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