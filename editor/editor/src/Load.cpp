#include <iostream>
#include <unordered_map>
#include <vector>

#include <core/Load.h>
#include <core/PoolAllocator.h>

#include "../include/EditorCameraSystem.h"

using namespace PhysicsEngine;

Asset *PhysicsEngine::loadAsset(WorldAllocators& allocators, WorldIdState& state, std::istream& in, const Guid& id, int type)
{
    return nullptr;
}

Component *PhysicsEngine::loadComponent(WorldAllocators& allocators, WorldIdState& state, std::istream& in, const Guid& id, int type)
{
    return nullptr;
}

System *PhysicsEngine::loadSystem(WorldAllocators& allocators, WorldIdState& state, std::istream& in, const Guid& id, int type)
{
    return nullptr;
}

Component *PhysicsEngine::destroyComponent(WorldAllocators& allocators, WorldIdState& state, const Guid& id, int type, int index)
{
    return nullptr;
}