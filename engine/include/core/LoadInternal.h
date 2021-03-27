#ifndef LOADINTERNAL_H__
#define LOADINTERNAL_H__

#include "WorldAllocators.h"
#include "WorldIdState.h"

namespace PhysicsEngine
{
// Get internal component
Component* getInternalComponent(const WorldAllocators& allocators, const WorldIdState& state, const Guid& id, int type);

// Load internal scene from YAML
Scene* loadInternalScene(WorldAllocators& allocators, WorldIdState& state, const YAML::Node& in, const Guid& id);

// Load internal entity either from binary or YAML
Entity *loadInternalEntity(WorldAllocators &allocators, WorldIdState &state, std::istream &in, const Guid &id);
Entity* loadInternalEntity(WorldAllocators& allocators, WorldIdState& state, const YAML::Node& in, const Guid& id);

// Load internal component either from binary or YAML
Component *loadInternalComponent(WorldAllocators &allocators, WorldIdState &state, std::istream &in, const Guid &id,
                                 int type);
Component* loadInternalComponent(WorldAllocators& allocators, WorldIdState& state, const YAML::Node& in, const Guid& id,
    int type);

// Load internal system from either binary or YAML
System *loadInternalSystem(WorldAllocators &allocators, WorldIdState &state, std::istream &in, const Guid &id,
                           int type);
System* loadInternalSystem(WorldAllocators& allocators, WorldIdState& state, const YAML::Node& in, const Guid& id,
    int type);

// Load internal asset from either binary YAML
Asset *loadInternalAsset(WorldAllocators &allocators, WorldIdState &state, std::istream &in, const Guid &id, int type);
Asset* loadInternalAsset(WorldAllocators& allocators, WorldIdState& state, const YAML::Node& in, const Guid& id, int type);

// Destroy internal entity, component, system or asset
Entity *destroyInternalEntity(WorldAllocators &allocators, WorldIdState &state, const Guid &id, int index);
Component *destroyInternalComponent(WorldAllocators &allocators, WorldIdState &state, const Guid &id, int type,
                                    int index);
} // namespace PhysicsEngine

#endif