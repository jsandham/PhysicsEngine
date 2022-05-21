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

    // internal world asset id state
    std::unordered_map<Guid, int> mMeshIdToGlobalIndex;
    std::unordered_map<Guid, int> mMaterialIdToGlobalIndex;
    std::unordered_map<Guid, int> mShaderIdToGlobalIndex;
    std::unordered_map<Guid, int> mTexture2DIdToGlobalIndex;
    std::unordered_map<Guid, int> mTexture3DIdToGlobalIndex;
    std::unordered_map<Guid, int> mCubemapIdToGlobalIndex;
    std::unordered_map<Guid, int> mRenderTextureIdToGlobalIndex;
    std::unordered_map<Guid, int> mFontIdToGlobalIndex;
    std::unordered_map<Guid, int> mSpriteIdToGlobalIndex;

    // internal world system id state
    std::unordered_map<Guid, int> mRenderSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> mPhysicsSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> mCleanupSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> mDebugSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> mGizmoSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> mFreeLookCameraSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> mTerrainSystemIdToGlobalIndex;

    // world id state for all scenes, systems, and assets
    std::unordered_map<Guid, int> mIdToGlobalIndex;
    std::unordered_map<Guid, int> mIdToType;

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