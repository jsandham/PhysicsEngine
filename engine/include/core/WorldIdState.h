#ifndef WORLD_ID_STATE_H__
#define WORLD_ID_STATE_H__

#include <unordered_map>

#include "Allocator.h"
#include "Guid.h"

namespace PhysicsEngine
{
// Simple structs used for grouping world id to global index maps when passing to functions
struct WorldIdState
{
    // internal world scene id state
    std::unordered_map<Guid, int> mSceneIdToGlobalIndex;

    // internal world entity id state
    std::unordered_map<Guid, int> mEntityIdToGlobalIndex;

    // internal world components id state
    std::unordered_map<Guid, int> mTransformIdToGlobalIndex;
    std::unordered_map<Guid, int> mMeshRendererIdToGlobalIndex;
    std::unordered_map<Guid, int> mLineRendererIdToGlobalIndex;
    std::unordered_map<Guid, int> mRigidbodyIdToGlobalIndex;
    std::unordered_map<Guid, int> mCameraIdToGlobalIndex;
    std::unordered_map<Guid, int> mLightIdToGlobalIndex;
    std::unordered_map<Guid, int> mSphereColliderIdToGlobalIndex;
    std::unordered_map<Guid, int> mBoxColliderIdToGlobalIndex;
    std::unordered_map<Guid, int> mCapsuleColliderIdToGlobalIndex;
    std::unordered_map<Guid, int> mMeshColliderIdToGlobalIndex;

    // internal world asset id state
    std::unordered_map<Guid, int> mMeshIdToGlobalIndex;
    std::unordered_map<Guid, int> mMaterialIdToGlobalIndex;
    std::unordered_map<Guid, int> mShaderIdToGlobalIndex;
    std::unordered_map<Guid, int> mTexture2DIdToGlobalIndex;
    std::unordered_map<Guid, int> mTexture3DIdToGlobalIndex;
    std::unordered_map<Guid, int> mCubemapIdToGlobalIndex;
    std::unordered_map<Guid, int> mFontIdToGlobalIndex;

    // internal world system id state
    std::unordered_map<Guid, int> mRenderSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> mPhysicsSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> mCleanupSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> mDebugSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> mGizmoSystemIdToGlobalIndex;

    // world id state for all entity, components, systems, and assets
    std::unordered_map<Guid, int> mIdToGlobalIndex;
    std::unordered_map<Guid, int> mIdToType;

    // entity ids to component ids
    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>> mEntityIdToComponentIds;

    // entity creation/deletion state
    std::vector<Guid> mEntityIdsMarkedCreated;
    std::vector<Guid> mEntityIdsMarkedLatentDestroy;
    std::vector<std::pair<Guid, int>> mEntityIdsMarkedMoved;

    // component create/deletion state
    std::vector<std::tuple<Guid, Guid, int>> mComponentIdsMarkedCreated;
    std::vector<std::tuple<Guid, Guid, int>> mComponentIdsMarkedLatentDestroy;
    std::vector<std::tuple<Guid, int, int>> mComponentIdsMarkedMoved;

    // asset create/deletion state
    std::vector<std::pair<Guid, int>> mAssetIdsMarkedCreated;
    std::vector<std::pair<Guid, int>> mAssetIdsMarkedLatentDestroy;
    std::vector<std::pair<Guid, int>> mAssetIdsMarkedMoved;

    // asset and scene id to filepath
    std::unordered_map<Guid, std::string> mAssetIdToFilepath;
    std::unordered_map<Guid, std::string> mSceneIdToFilepath;

    // asset and scene filepath to id
    std::unordered_map<std::string, Guid> mAssetFilepathToId;
    std::unordered_map<std::string, Guid> mSceneFilepathToId;
};
} // namespace PhysicsEngine

#endif