#include <iostream>

#include "../../include/systems/DebugSystem.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"

#include "../../include/components/MeshRenderer.h"
#include "../../include/components/Transform.h"

using namespace PhysicsEngine;

DebugSystem::DebugSystem(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
}

DebugSystem::DebugSystem(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
}

void DebugSystem::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;
}

void DebugSystem::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");
}

int DebugSystem::getType() const
{
    return PhysicsEngine::DEBUGSYSTEM_TYPE;
}

std::string DebugSystem::getObjectName() const
{
    return PhysicsEngine::DEBUGSYSTEM_NAME;
}

Guid DebugSystem::getGuid() const
{
    return mGuid;
}

Id DebugSystem::getId() const
{
    return mId;
}

void DebugSystem::init(World *world)
{
    mWorld = world;
}

void DebugSystem::update()
{
}