#include <iostream>
#include <unordered_map>
#include <vector>

#include <core/Load.h>
#include <core/PoolAllocator.h>

#include "../include/EditorCameraSystem.h"

using namespace PhysicsEngine;

Component* PhysicsEngine::getComponent(const WorldAllocators& allocators, const WorldIdState& state, const Guid& id, int type)
{
    return nullptr;
}

Asset* PhysicsEngine::getAsset(const WorldAllocators& allocators, const WorldIdState& state, const Guid& id, int type)
{
    return nullptr;
}

Asset* PhysicsEngine::loadAsset(World& world, WorldAllocators& allocators, WorldIdState& state, const YAML::Node& in, const Guid& id, int type)
{
    return nullptr;
}

Component* PhysicsEngine::loadComponent(World& world, WorldAllocators& allocators, WorldIdState& state, const YAML::Node& in, const Guid& id, int type)
{
    return nullptr;
}

System* PhysicsEngine::loadSystem(World& world, WorldAllocators& allocators, WorldIdState& state, const YAML::Node& in, const Guid& id, int type)
{
    return nullptr;
}

Component *PhysicsEngine::destroyComponent(WorldAllocators& allocators, WorldIdState& state, const Guid& entityId, const Guid& componentId, int type, int index)
{
    return nullptr;
}

Asset* PhysicsEngine::destroyAsset(WorldAllocators& allocators, WorldIdState& state, const Guid& assetId, int type, int index)
{
    return nullptr;
}