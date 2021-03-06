#ifndef LOADINTERNAL_H__
#define LOADINTERNAL_H__

#include <unordered_map>
#include <vector>

#include "Allocator.h"
#include "PoolAllocator.h"

#include "Entity.h"

#include "Asset.h"
#include "Cubemap.h"
#include "Font.h"
#include "Material.h"
#include "Mesh.h"
#include "Shader.h"
#include "Texture2D.h"
#include "Texture3D.h"
#include "WorldAllocators.h"
#include "WorldIdState.h"

#include "../components/BoxCollider.h"
#include "../components/Camera.h"
#include "../components/CapsuleCollider.h"
#include "../components/Component.h"
#include "../components/Light.h"
#include "../components/LineRenderer.h"
#include "../components/MeshCollider.h"
#include "../components/MeshRenderer.h"
#include "../components/Rigidbody.h"
#include "../components/SphereCollider.h"
#include "../components/Transform.h"

#include "../systems/CleanUpSystem.h"
#include "../systems/DebugSystem.h"
#include "../systems/GizmoSystem.h"
#include "../systems/PhysicsSystem.h"
#include "../systems/RenderSystem.h"
#include "../systems/System.h"

namespace PhysicsEngine
{
// Get internal component
Component* getInternalComponent(WorldAllocators& allocators, WorldIdState& state, const Guid& id, int type);

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