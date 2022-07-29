#ifndef WORLD_ID_STATE_H__
#define WORLD_ID_STATE_H__

namespace PhysicsEngine
{
// Simple structs used for grouping world id to global index maps when passing to functions
struct WorldIdState
{
    // world scene guid state
    std::unordered_map<Guid, int> mSceneGuidToGlobalIndex;

    // world asset guid state
    std::unordered_map<Guid, int> mMeshGuidToGlobalIndex;
    std::unordered_map<Guid, int> mMaterialGuidToGlobalIndex;
    std::unordered_map<Guid, int> mShaderGuidToGlobalIndex;
    std::unordered_map<Guid, int> mTexture2DGuidToGlobalIndex;
    std::unordered_map<Guid, int> mTexture3DGuidToGlobalIndex;
    std::unordered_map<Guid, int> mCubemapGuidToGlobalIndex;
    std::unordered_map<Guid, int> mRenderTextureGuidToGlobalIndex;
    std::unordered_map<Guid, int> mFontGuidToGlobalIndex;
    std::unordered_map<Guid, int> mSpriteGuidToGlobalIndex;

    // world system guid state
    std::unordered_map<Guid, int> mRenderSystemGuidToGlobalIndex;
    std::unordered_map<Guid, int> mPhysicsSystemGuidToGlobalIndex;
    std::unordered_map<Guid, int> mCleanupSystemGuidToGlobalIndex;
    std::unordered_map<Guid, int> mDebugSystemGuidToGlobalIndex;
    std::unordered_map<Guid, int> mGizmoSystemGuidToGlobalIndex;
    std::unordered_map<Guid, int> mFreeLookCameraSystemGuidToGlobalIndex;
    std::unordered_map<Guid, int> mTerrainSystemGuidToGlobalIndex;

    // world id state for all scenes, systems, and assets
    std::unordered_map<Guid, int> mGuidToGlobalIndex;
    std::unordered_map<Guid, int> mGuidToType;

    // asset create/deletion state
    std::vector<std::pair<Guid, int>> mAssetGuidsMarkedCreated;
    std::vector<std::pair<Guid, int>> mAssetGuidsMarkedLatentDestroy;
    std::vector<std::pair<Guid, int>> mAssetGuidsMarkedMoved;

    // asset and scene id to filepath
    std::unordered_map<Guid, std::string> mAssetGuidToFilepath;
    std::unordered_map<Guid, std::string> mSceneGuidToFilepath;

    // asset and scene filepath to id
    std::unordered_map<std::string, Guid> mAssetFilepathToGuid;
    std::unordered_map<std::string, Guid> mSceneFilepathToGuid;











    // world scene id state
    std::unordered_map<Id, int> mSceneIdToGlobalIndex;

    // world asset id state
    std::unordered_map<Id, int> mMeshIdToGlobalIndex;
    std::unordered_map<Id, int> mMaterialIdToGlobalIndex;
    std::unordered_map<Id, int> mShaderIdToGlobalIndex;
    std::unordered_map<Id, int> mTexture2DIdToGlobalIndex;
    std::unordered_map<Id, int> mTexture3DIdToGlobalIndex;
    std::unordered_map<Id, int> mCubemapIdToGlobalIndex;
    std::unordered_map<Id, int> mRenderTextureIdToGlobalIndex;
    std::unordered_map<Id, int> mFontIdToGlobalIndex;
    std::unordered_map<Id, int> mSpriteIdToGlobalIndex;

    // world system id state
    std::unordered_map<Id, int> mRenderSystemIdToGlobalIndex;
    std::unordered_map<Id, int> mPhysicsSystemIdToGlobalIndex;
    std::unordered_map<Id, int> mCleanupSystemIdToGlobalIndex;
    std::unordered_map<Id, int> mDebugSystemIdToGlobalIndex;
    std::unordered_map<Id, int> mGizmoSystemIdToGlobalIndex;
    std::unordered_map<Id, int> mFreeLookCameraSystemIdToGlobalIndex;
    std::unordered_map<Id, int> mTerrainSystemIdToGlobalIndex;

    // world id state for all scenes, systems, and assets
    std::unordered_map<Id, int> mIdToGlobalIndex;
    std::unordered_map<Id, int> mIdToType;

    std::unordered_map<Guid, Id> mGuidToId;
    std::unordered_map<Id, Guid> mIdToGuid;
};
} // namespace PhysicsEngine

#endif