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
// load external asset defined by the user (assets with type 10 or greater)
Asset *loadAsset(WorldAllocators &allocators, WorldIdState &state, std::istream &in, const Guid &id, int type);

// load external components defined by the user (components with type 20 or greater)
Component *loadComponent(WorldAllocators &allocators, WorldIdState &state, std::istream &in, const Guid &id, int type);

// load external systems defined by the user (systems with type 10 or greater)
System *loadSystem(WorldAllocators &allocators, WorldIdState &state, std::istream &in, const Guid &id, int type);

// destroy external components defined by the user
Component *destroyComponent(WorldAllocators &allocators, WorldIdState &state, const Guid &id, int type, int index);
} // namespace PhysicsEngine

#endif