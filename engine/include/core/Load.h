#ifndef LOAD_H__
#define LOAD_H__

#include <unordered_map>
#include <vector>

#include "../components/Component.h"
#include "../systems/System.h"
#include "Allocator.h"
#include "Asset.h"
#include "WorldAllocators.h"
#include "WorldIdState.h"

namespace PhysicsEngine
{
// get external component defined by the user
Component *getComponent(const WorldAllocators &allocators, const WorldIdState &state, const Guid &id, int type);

// get external asset defined by the user
Asset *getAsset(const WorldAllocators &allocators, const WorldIdState &state, const Guid &id, int type);

// load external asset defined by the user from YAML
Asset *loadAsset(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in, const Guid &id,
                 int type);

// load external components defined by the user from YAML
Component *loadComponent(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in,
                         const Guid &id, int type);

// load external systems defined by the user from YAML
System *loadSystem(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in, const Guid &id,
                   int type);

// destroy external components defined by the user
Component *destroyComponent(WorldAllocators &allocators, WorldIdState &state, const Guid &entityId,
                            const Guid &componentId, int type, int index);

// destroy external assets defined by the user
Asset *destroyAsset(WorldAllocators &allocators, WorldIdState &state, const Guid &assetId, int type, int index);
} // namespace PhysicsEngine

#endif