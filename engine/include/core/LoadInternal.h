#ifndef __LOADINTERNAL_H__
#define __LOADINTERNAL_H__

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
#include "../systems/PhysicsSystem.h"
#include "../systems/RenderSystem.h"
#include "../systems/GizmoSystem.h"
#include "../systems/System.h"

namespace PhysicsEngine
{
// Add internal entity, component, system or asset id to global index maps
void addInternalEntityIdToIndexMap(std::unordered_map<Guid, int> *entityIdToGlobalIndex,
                                   std::unordered_map<Guid, int> *idToGlobalIndex,
                                   std::unordered_map<Guid, int> *idToType, const Guid &id, int index);
void addInternalComponentIdToIndexMap(
    std::unordered_map<Guid, int> *transformIdToGlobalIndex, std::unordered_map<Guid, int> *meshRendererIdToGlobalIndex,
    std::unordered_map<Guid, int> *lineRendererIdToGlobalIndex, std::unordered_map<Guid, int> *rigidbodyIdToGlobalIndex,
    std::unordered_map<Guid, int> *cameraIdToGlobalIndex, std::unordered_map<Guid, int> *lightIdToGlobalIndex,
    std::unordered_map<Guid, int> *sphereColliderIdToGlobalIndex,
    std::unordered_map<Guid, int> *boxColliderIdToGlobalIndex,
    std::unordered_map<Guid, int> *capsuleColliderIdToGlobalIndex,
    std::unordered_map<Guid, int> *meshColliderIdToGlobalIndex, std::unordered_map<Guid, int> *idToGlobalIndex,
    std::unordered_map<Guid, int> *idToType, const Guid &id, int type, int index);
void addInternalSystemIdToIndexMap(std::unordered_map<Guid, int> *renderSystemIdToGlobalIndex,
                                   std::unordered_map<Guid, int> *physicsSystemIdToGlobalIndex,
                                   std::unordered_map<Guid, int> *cleanupSystemIdToGlobalIndex,
                                   std::unordered_map<Guid, int> *debugSystemIdToGlobalIndex,
                                   std::unordered_map<Guid, int>* gizmoSystemIdToGlobalIndex,
                                   std::unordered_map<Guid, int> *idToGlobalIndex,
                                   std::unordered_map<Guid, int> *idToType, const Guid &id, int type, int index);
void addInternalAssetIdToIndexMap(
    std::unordered_map<Guid, int> *meshIdToGlobalIndex, std::unordered_map<Guid, int> *materialIdToGlobalIndex,
    std::unordered_map<Guid, int> *shaderIdToGlobalIndex, std::unordered_map<Guid, int> *texture2DIdToGlobalIndex,
    std::unordered_map<Guid, int> *texture3DIdToGlobalIndex, std::unordered_map<Guid, int> *cubemapIdToGlobalIndex,
    std::unordered_map<Guid, int> *fontIdToGlobalIndex, std::unordered_map<Guid, int> *idToGlobalIndex,
    std::unordered_map<Guid, int> *idToType, const Guid &id, int type, int index);

// Remove internal entity, component, system or asset id to global index maps
void removeInternalEntityIdFromIndexMap(std::unordered_map<Guid, int> *entityIdToGlobalIndex,
                                        std::unordered_map<Guid, int> *idToGlobalIndex,
                                        std::unordered_map<Guid, int> *idToType, const Guid &id);
void removeInternalComponentIdFromIndexMap(
    std::unordered_map<Guid, int> *transformIdToGlobalIndex, std::unordered_map<Guid, int> *meshRendererIdToGlobalIndex,
    std::unordered_map<Guid, int> *lineRendererIdToGlobalIndex, std::unordered_map<Guid, int> *rigidbodyIdToGlobalIndex,
    std::unordered_map<Guid, int> *cameraIdToGlobalIndex, std::unordered_map<Guid, int> *lightIdToGlobalIndex,
    std::unordered_map<Guid, int> *sphereColliderIdToGlobalIndex,
    std::unordered_map<Guid, int> *boxColliderIdToGlobalIndex,
    std::unordered_map<Guid, int> *capsuleColliderIdToGlobalIndex,
    std::unordered_map<Guid, int> *meshColliderIdToGlobalIndex, std::unordered_map<Guid, int> *idToGlobalIndex,
    std::unordered_map<Guid, int> *idToType, const Guid &id, int type);
void removeInternalSystemIdFromIndexMap(std::unordered_map<Guid, int> *renderSystemIdToGlobalIndex,
                                        std::unordered_map<Guid, int> *physicsSystemIdToGlobalIndex,
                                        std::unordered_map<Guid, int> *cleanupSystemIdToGlobalIndex,
                                        std::unordered_map<Guid, int> *debugSystemIdToGlobalIndex,
                                        std::unordered_map<Guid, int>* gizmoSystemIdToGlobalIndex,
                                        std::unordered_map<Guid, int> *idToGlobalIndex,
                                        std::unordered_map<Guid, int> *idToType, const Guid &id, int type);
void removeInternalAssetIdFromIndexMap(
    std::unordered_map<Guid, int> *meshIdToGlobalIndex, std::unordered_map<Guid, int> *materialIdToGlobalIndex,
    std::unordered_map<Guid, int> *shaderIdToGlobalIndex, std::unordered_map<Guid, int> *texture2DIdToGlobalIndex,
    std::unordered_map<Guid, int> *texture3DIdToGlobalIndex, std::unordered_map<Guid, int> *cubemapIdToGlobalIndex,
    std::unordered_map<Guid, int> *fontIdToGlobalIndex, std::unordered_map<Guid, int> *idToGlobalIndex,
    std::unordered_map<Guid, int> *idToType, const Guid &id, int type);

// Get internal entity, component, system or asset from allocators
Entity *getInternalEntity(PoolAllocator<Entity> *entityAllocator, int index);
Component *getInternalComponent(PoolAllocator<Transform> *transformAllocator,
                                PoolAllocator<MeshRenderer> *meshRendererAllocator,
                                PoolAllocator<LineRenderer> *lineRendererAllocator,
                                PoolAllocator<Rigidbody> *rigidbodyAllocator, PoolAllocator<Camera> *cameraAllocator,
                                PoolAllocator<Light> *lightAllocator,
                                PoolAllocator<SphereCollider> *sphereColliderAllocator,
                                PoolAllocator<BoxCollider> *boxColliderAllocator,
                                PoolAllocator<CapsuleCollider> *capsuleColliderAllocator,
                                PoolAllocator<MeshCollider> *meshColliderAllocator, int type, int index);
System *getInternalSystem(PoolAllocator<RenderSystem> *renderSystemAllocator,
                          PoolAllocator<PhysicsSystem> *physicsSystemAllocator,
                          PoolAllocator<CleanUpSystem> *cleanupSystemAllocator,
                          PoolAllocator<DebugSystem> *debugSystemAllocator,
                          PoolAllocator<GizmoSystem>* gizmoSystemAllocator, int type, int index);
Asset *getInternalAsset(PoolAllocator<Mesh> *meshAllocator, PoolAllocator<Material> *materialAllocator,
                        PoolAllocator<Shader> *shaderAllocator, PoolAllocator<Texture2D> *texture2DAllocator,
                        PoolAllocator<Texture3D> *texture3DAllocator, PoolAllocator<Cubemap> *cubemapAllocator,
                        PoolAllocator<Font> *fontAllocator, int type, int index);

// Load internal entity, component, system or asset into allocators
Entity *loadInternalEntity(PoolAllocator<Entity> *entityAllocator, const std::vector<char> &data, int *index);
Component *loadInternalComponent(
    PoolAllocator<Transform> *transformAllocator, PoolAllocator<MeshRenderer> *meshRendererAllocator,
    PoolAllocator<LineRenderer> *lineRendererAllocator, PoolAllocator<Rigidbody> *rigidbodyAllocator,
    PoolAllocator<Camera> *cameraAllocator, PoolAllocator<Light> *lightAllocator,
    PoolAllocator<SphereCollider> *sphereColliderAllocator, PoolAllocator<BoxCollider> *boxColliderAllocator,
    PoolAllocator<CapsuleCollider> *capsuleColliderAllocator, PoolAllocator<MeshCollider> *meshColliderAllocator,
    const std::vector<char> &data, int type, int *index);
System *loadInternalSystem(PoolAllocator<RenderSystem> *renderSystemAllocator,
                           PoolAllocator<PhysicsSystem> *physicsSystemAllocator,
                           PoolAllocator<CleanUpSystem> *cleanupSystemAllocator,
                           PoolAllocator<DebugSystem> *debugSystemAllocator,
                           PoolAllocator<GizmoSystem>* gizmoSystemAllocator, const std::vector<char> &data, int type,
                           int *index);
Asset *loadInternalAsset(PoolAllocator<Mesh> *meshAllocator, PoolAllocator<Material> *materialAllocator,
                         PoolAllocator<Shader> *shaderAllocator, PoolAllocator<Texture2D> *texture2DAllocator,
                         PoolAllocator<Texture3D> *texture3DAllocator, PoolAllocator<Cubemap> *cubemapAllocator,
                         PoolAllocator<Font> *fontAllocator, const std::vector<char> &data, int type, int *index);

// Destroy internal entity, component, system or asset into allocators
Entity *destroyInternalEntity(PoolAllocator<Entity> *entityAllocator, int index);
Component *destroyInternalComponent(PoolAllocator<Transform> *transformAllocator,
                                    PoolAllocator<MeshRenderer> *meshRendererAllocator,
                                    PoolAllocator<LineRenderer> *lineRendererAllocator,
                                    PoolAllocator<Rigidbody> *rigidbodyAllocator,
                                    PoolAllocator<Camera> *cameraAllocator, PoolAllocator<Light> *lightAllocator,
                                    PoolAllocator<SphereCollider> *sphereColliderAllocator,
                                    PoolAllocator<BoxCollider> *boxColliderAllocator,
                                    PoolAllocator<CapsuleCollider> *capsuleColliderAllocator,
                                    PoolAllocator<MeshCollider> *meshColliderAllocator, int type, int index);
} // namespace PhysicsEngine

#endif