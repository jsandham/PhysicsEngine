
#ifndef __WORLD_UTIL_H__
#define __WORLD_UTIL_H__

#include <unordered_map>

#include "Allocator.h"
#include "Guid.h"
#include "PoolAllocator.h"

namespace PhysicsEngine
{
// Simple structs used for grouping world allocators when passing to functions
struct WorldAllocators
{
    // internal entity allocator
    PoolAllocator<Entity> *mEntityAllocator;

    // internal component allocators
    PoolAllocator<Transform> *mTransformAllocator;
    PoolAllocator<MeshRenderer> *mMeshRendererAllocator;
    PoolAllocator<LineRenderer> *mLineRendererAllocator;
    PoolAllocator<Rigidbody> *mRigidbodyAllocator;
    PoolAllocator<Camera> *mCameraAllocator;
    PoolAllocator<Light> *mLightAllocator;
    PoolAllocator<SphereCollider> *mSphereColliderAllocator;
    PoolAllocator<BoxCollider> *mBoxColliderAllocator;
    PoolAllocator<CapsuleCollider> *mCapsuleColliderAllocator;
    PoolAllocator<MeshCollider> *mMeshColliderAllocator;

    // internal asset allocators
    PoolAllocator<Mesh> *mMeshAllocator;
    PoolAllocator<Material> *mMaterialAllocator;
    PoolAllocator<Shader> *mShaderAllocator;
    PoolAllocator<Texture2D> *mTexture2DAllocator;
    PoolAllocator<Texture3D> *mTexture3DAllocator;
    PoolAllocator<Cubemap> *mCubemapAllocator;
    PoolAllocator<Font> *mFontAllocator;

    // internal system allocators
    PoolAllocator<RenderSystem> *mRenderSystemAllocator;
    PoolAllocator<PhysicsSystem> *mPhysicsSystemAllocator;
    PoolAllocator<CleanUpSystem> *mCleanupSystemAllocator;
    PoolAllocator<DebugSystem> *mDebugSystemAllocator;
    PoolAllocator<GizmoSystem> *mGizmoSystemAllocator;

    // non-internal allocators for user defined components, systems and assets
    std::unordered_map<int, Allocator *> *mComponentAllocatorMap;
    std::unordered_map<int, Allocator *> *mSystemAllocatorMap;
    std::unordered_map<int, Allocator *> *mAssetAllocatorMap;
};

// Simple structs used for grouping world id to global index maps when passing to functions
struct WorldIsState
{
    // internal world entity id state
    std::unordered_map<Guid, int> *mEntityIdToGlobalIndex;

    // internal world components id state
    std::unordered_map<Guid, int> *mTransformIdToGlobalIndex;
    std::unordered_map<Guid, int> *mMeshRendererIdToGlobalIndex;
    std::unordered_map<Guid, int> *mLineRendererIdToGlobalIndex;
    std::unordered_map<Guid, int> *mRigidbodyIdToGlobalIndex;
    std::unordered_map<Guid, int> *mCameraIdToGlobalIndex;
    std::unordered_map<Guid, int> *mLightIdToGlobalIndex;
    std::unordered_map<Guid, int> *mSphereColliderIdToGlobalIndex;
    std::unordered_map<Guid, int> *mBoxColliderIdToGlobalIndex;
    std::unordered_map<Guid, int> *mCapsuleColliderIdToGlobalIndex;
    std::unordered_map<Guid, int> *mMeshColliderIdToGlobalIndex;

    // internal world asset id state
    std::unordered_map<Guid, int> *mMeshIdToGlobalIndex;
    std::unordered_map<Guid, int> *mMaterialIdToGlobalIndex;
    std::unordered_map<Guid, int> *mShaderIdToGlobalIndex;
    std::unordered_map<Guid, int> *mTexture2DIdToGlobalIndex;
    std::unordered_map<Guid, int> *mTexture3DIdToGlobalIndex;
    std::unordered_map<Guid, int> *mCubemapIdToGlobalIndex;
    std::unordered_map<Guid, int> *mFontIdToGlobalIndex;

    // internal world system id state
    std::unordered_map<Guid, int> *mRenderSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> *mPhysicsSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> *mCleanupSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> *mDebugSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> *mGizmoSystemIdToGlobalIndex;

    // world id state for all entity, components, systems, and assets
    std::unordered_map<Guid, int> *mIdToGlobalIndex;
    std::unordered_map<Guid, int> *mIdToType;

    // entity ids to component ids
    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>> *mEntityIdToComponentIds;

    // entity creation/deletion state
    std::vector<Guid> *mEntityIdsMarkedCreated;
    std::vector<Guid> *mEntityIdsMarkedLatentDestroy;
    std::vector<std::pair<Guid, int>> *mEntityIdsMarkedMoved;

    // component create/deletion state
    std::vector<triple<Guid, Guid, int>> *mComponentIdsMarkedCreated;
    std::vector<triple<Guid, Guid, int>> *mComponentIdsMarkedLatentDestroy;
    std::vector<triple<Guid, int, int>> *mComponentIdsMarkedMoved;
};
} // namespace PhysicsEngine

#endif