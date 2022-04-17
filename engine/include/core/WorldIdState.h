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
    std::unordered_map<Id, int> mSceneIdToGlobalIndex;

    // internal world entity id state
    std::unordered_map<Id, int> mEntityIdToGlobalIndex;

    // internal world components id state
    std::unordered_map<Id, int> mTransformIdToGlobalIndex;
    std::unordered_map<Id, int> mMeshRendererIdToGlobalIndex;
    std::unordered_map<Id, int> mSpriteRendererIdToGlobalIndex;
    std::unordered_map<Id, int> mLineRendererIdToGlobalIndex;
    std::unordered_map<Id, int> mRigidbodyIdToGlobalIndex;
    std::unordered_map<Id, int> mCameraIdToGlobalIndex;
    std::unordered_map<Id, int> mLightIdToGlobalIndex;
    std::unordered_map<Id, int> mSphereColliderIdToGlobalIndex;
    std::unordered_map<Id, int> mBoxColliderIdToGlobalIndex;
    std::unordered_map<Id, int> mCapsuleColliderIdToGlobalIndex;
    std::unordered_map<Id, int> mMeshColliderIdToGlobalIndex;
    std::unordered_map<Id, int> mTerrainIdToGlobalIndex;

    // internal world asset id state
    std::unordered_map<Id, int> mMeshIdToGlobalIndex;
    std::unordered_map<Id, int> mMaterialIdToGlobalIndex;
    std::unordered_map<Id, int> mShaderIdToGlobalIndex;
    std::unordered_map<Id, int> mTexture2DIdToGlobalIndex;
    std::unordered_map<Id, int> mTexture3DIdToGlobalIndex;
    std::unordered_map<Id, int> mCubemapIdToGlobalIndex;
    std::unordered_map<Id, int> mRenderTextureIdToGlobalIndex;
    std::unordered_map<Id, int> mFontIdToGlobalIndex;
    std::unordered_map<Id, int> mSpriteIdToGlobalIndex;

    // internal world system id state
    std::unordered_map<Id, int> mRenderSystemIdToGlobalIndex;
    std::unordered_map<Id, int> mPhysicsSystemIdToGlobalIndex;
    std::unordered_map<Id, int> mCleanupSystemIdToGlobalIndex;
    std::unordered_map<Id, int> mDebugSystemIdToGlobalIndex;
    std::unordered_map<Id, int> mGizmoSystemIdToGlobalIndex;
    std::unordered_map<Id, int> mFreeLookCameraSystemIdToGlobalIndex;
    std::unordered_map<Id, int> mTerrainSystemIdToGlobalIndex;

    // world id state for all entity, components, systems, and assets
    std::unordered_map<Guid, Id> mGuidToId;
    std::unordered_map<Id, Guid> mIdToGuid;
    std::unordered_map<Id, int> mIdToGlobalIndex;
    std::unordered_map<Id, int> mIdToType;

    // entity ids to component ids
    std::unordered_map<Id, std::vector<std::pair<Id, int>>> mEntityIdToComponentIds;

    // entity creation/deletion state
    std::vector<Id> mEntityIdsMarkedCreated;
    std::vector<Id> mEntityIdsMarkedLatentDestroy;
    std::vector<std::pair<Id, int>> mEntityIdsMarkedMoved;

    // component create/deletion state
    std::vector<std::tuple<Id, Id, int>> mComponentIdsMarkedCreated;
    std::vector<std::tuple<Id, Id, int>> mComponentIdsMarkedLatentDestroy;
    std::vector<std::tuple<Id, int, int>> mComponentIdsMarkedMoved;

    // asset create/deletion state
    std::vector<std::pair<Id, int>> mAssetIdsMarkedCreated;
    std::vector<std::pair<Id, int>> mAssetIdsMarkedLatentDestroy;
    std::vector<std::pair<Id, int>> mAssetIdsMarkedMoved;

    // asset and scene id to filepath
    std::unordered_map<Guid, std::string> mAssetIdToFilepath;
    std::unordered_map<Guid, std::string> mSceneIdToFilepath;

    // asset and scene filepath to id
    std::unordered_map<std::string, Guid> mAssetFilepathToId;
    std::unordered_map<std::string, Guid> mSceneFilepathToId;
};
} // namespace PhysicsEngine

#endif