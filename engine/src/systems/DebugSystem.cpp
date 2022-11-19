#include <iostream>

#include "../../include/systems/DebugSystem.h"

#include "../../include/core/Input.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"

#include "../../include/components/MeshRenderer.h"
#include "../../include/components/Transform.h"

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"

using namespace PhysicsEngine;

DebugSystem::DebugSystem(World *world, const Id &id) : System(world, id)
{
}

DebugSystem::DebugSystem(World *world, const Guid &guid, const Id &id) : System(world, guid, id)
{
}

DebugSystem::~DebugSystem()
{
}

void DebugSystem::serialize(YAML::Node &out) const
{
    System::serialize(out);
}

void DebugSystem::deserialize(const YAML::Node &in)
{
    System::deserialize(in);
}

int DebugSystem::getType() const
{
    return PhysicsEngine::DEBUGSYSTEM_TYPE;
}

std::string DebugSystem::getObjectName() const
{
    return PhysicsEngine::DEBUGSYSTEM_NAME;
}

void DebugSystem::init(World *world)
{
    mWorld = world;
}

void DebugSystem::update(const Input &input, const Time &time)
{
}