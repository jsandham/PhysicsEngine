#ifndef LOADINTERNAL_H__
#define LOADINTERNAL_H__

#include "WorldAllocators.h"
#include "WorldIdState.h"

namespace PhysicsEngine
{
// Get internal component
Component *getInternalComponent(const WorldAllocators &allocators, const WorldIdState &state, Id id, int type);

// Get internal asset
Asset *getInternalAsset(const WorldAllocators &allocators, const WorldIdState &state, Id id, int type);

//// Load internal scene from YAML
//Scene *loadInternalScene(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in, Id id);
//
//// Load internal entity either from binary or YAML
//Entity *loadInternalEntity(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in, Id id);
//
//// Load internal component either from binary or YAML
//Component *loadInternalComponent(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in, Id id, int type);
//
//// Load internal system from either binary or YAML
//System *loadInternalSystem(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in,
//                           Id id, int type);
//
//// Load internal asset from either binary YAML
//Asset *loadInternalAsset(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in,
//                         Id id, int type);

// Destroy internal entity, component, system or asset
Entity *destroyInternalEntity(WorldAllocators &allocators, WorldIdState &state, Id entityId, int index);
Component *destroyInternalComponent(WorldAllocators &allocators, WorldIdState &state, Id entityId,
                                    Id componentId, int type, int index);
Asset *destroyInternalAsset(WorldAllocators &allocators, WorldIdState &state, Id assetId, int type, int index);
} // namespace PhysicsEngine

#endif