#include <fstream>
#include <stack>
#include <assert.h>

#include "../../include/core/Log.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

template <> size_t World::getNumberOfSystems<RenderSystem>() const
{
    return mAllocators.mRenderSystemAllocator.getCount();
}

template <> size_t World::getNumberOfSystems<PhysicsSystem>() const
{
    return mAllocators.mPhysicsSystemAllocator.getCount();
}
template <> size_t World::getNumberOfSystems<CleanUpSystem>() const
{
    return mAllocators.mCleanupSystemAllocator.getCount();
}

template <> size_t World::getNumberOfSystems<DebugSystem>() const
{
    return mAllocators.mDebugSystemAllocator.getCount();
}

template <> size_t World::getNumberOfSystems<GizmoSystem>() const
{
    return mAllocators.mGizmoSystemAllocator.getCount();
}

template <> size_t World::getNumberOfSystems<FreeLookCameraSystem>() const
{
    return mAllocators.mFreeLookCameraSystemAllocator.getCount();
}

template <> size_t World::getNumberOfSystems<TerrainSystem>() const
{
    return mAllocators.mTerrainSystemAllocator.getCount();
}

template <> size_t World::getNumberOfSystems<AssetLoadingSystem>() const
{
    return mAllocators.mAssetLoadingSystemAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<Mesh>() const
{
    return mAllocators.mMeshAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<Material>() const
{
    return mAllocators.mMaterialAllocator.getCount();
}
template <> size_t World::getNumberOfAssets<Shader>() const
{
    return mAllocators.mShaderAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<Texture2D>() const
{
    return mAllocators.mTexture2DAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<Cubemap>() const
{
    return mAllocators.mCubemapAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<RenderTexture>() const
{
    return mAllocators.mRenderTextureAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<Font>() const
{
    return mAllocators.mFontAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<Sprite>() const
{
    return mAllocators.mSpriteAllocator.getCount();
}

template <> RenderSystem* World::getSystem<RenderSystem>() const
{
    return mAllocators.mRenderSystemAllocator.get(0);
}

template <> PhysicsSystem* World::getSystem<PhysicsSystem>() const
{
    return mAllocators.mPhysicsSystemAllocator.get(0);
}
template <> CleanUpSystem* World::getSystem<CleanUpSystem>() const
{
    return mAllocators.mCleanupSystemAllocator.get(0);
}

template <> DebugSystem* World::getSystem<DebugSystem>() const
{
    return mAllocators.mDebugSystemAllocator.get(0);
}

template <> GizmoSystem* World::getSystem<GizmoSystem>() const
{
    return mAllocators.mGizmoSystemAllocator.get(0);
}

template <> FreeLookCameraSystem* World::getSystem<FreeLookCameraSystem>() const
{
    return mAllocators.mFreeLookCameraSystemAllocator.get(0);
}

template <> TerrainSystem* World::getSystem<TerrainSystem>() const
{
    return mAllocators.mTerrainSystemAllocator.get(0);
}

template <> AssetLoadingSystem* World::getSystem<AssetLoadingSystem>() const
{
    return mAllocators.mAssetLoadingSystemAllocator.get(0);
}

template <> RenderSystem* World::addSystem<RenderSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mRenderSystemAllocator, order);
}

template <> PhysicsSystem* World::addSystem<PhysicsSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mPhysicsSystemAllocator, order);
}
template <> CleanUpSystem* World::addSystem<CleanUpSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mCleanupSystemAllocator, order);
}

template <> DebugSystem* World::addSystem<DebugSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mDebugSystemAllocator, order);
}

template <> GizmoSystem* World::addSystem<GizmoSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mGizmoSystemAllocator, order);
}

template <> FreeLookCameraSystem* World::addSystem<FreeLookCameraSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mFreeLookCameraSystemAllocator, order);
}

template <> TerrainSystem* World::addSystem<TerrainSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mTerrainSystemAllocator, order);
}

template <> AssetLoadingSystem* World::addSystem<AssetLoadingSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mAssetLoadingSystemAllocator, order);
}

template <> RenderSystem* World::getSystemByIndex<RenderSystem>(size_t index) const
{
    return mAllocators.mRenderSystemAllocator.get(index);
}

template <> PhysicsSystem* World::getSystemByIndex<PhysicsSystem>(size_t index) const
{
    return mAllocators.mPhysicsSystemAllocator.get(index);
}
template <> CleanUpSystem* World::getSystemByIndex<CleanUpSystem>(size_t index) const
{
    return mAllocators.mCleanupSystemAllocator.get(index);
}

template <> DebugSystem* World::getSystemByIndex<DebugSystem>(size_t index) const
{
    return mAllocators.mDebugSystemAllocator.get(index);
}

template <> GizmoSystem* World::getSystemByIndex<GizmoSystem>(size_t index) const
{
    return mAllocators.mGizmoSystemAllocator.get(index);
}

template <> FreeLookCameraSystem* World::getSystemByIndex<FreeLookCameraSystem>(size_t index) const
{
    return mAllocators.mFreeLookCameraSystemAllocator.get(index);
}

template <> TerrainSystem* World::getSystemByIndex<TerrainSystem>(size_t index) const
{
    return mAllocators.mTerrainSystemAllocator.get(index);
}

template <> AssetLoadingSystem* World::getSystemByIndex<AssetLoadingSystem>(size_t index) const
{
    return mAllocators.mAssetLoadingSystemAllocator.get(index);
}

template <> RenderSystem* World::getSystemById<RenderSystem>(const Id& systemId) const
{
    return getSystemById_impl(mIdState.mRenderSystemIdToGlobalIndex, &mAllocators.mRenderSystemAllocator, systemId);
}

template <> PhysicsSystem *World::getSystemById<PhysicsSystem>(const Id &systemId) const
{
    return getSystemById_impl(mIdState.mPhysicsSystemIdToGlobalIndex, &mAllocators.mPhysicsSystemAllocator,
        systemId);
}
template <> CleanUpSystem *World::getSystemById<CleanUpSystem>(const Id &systemId) const
{
    return getSystemById_impl(mIdState.mCleanupSystemIdToGlobalIndex, &mAllocators.mCleanupSystemAllocator,
        systemId);
}

template <> DebugSystem *World::getSystemById<DebugSystem>(const Id &systemId) const
{
    return getSystemById_impl(mIdState.mDebugSystemIdToGlobalIndex, &mAllocators.mDebugSystemAllocator, systemId);
}

template <> GizmoSystem *World::getSystemById<GizmoSystem>(const Id &systemId) const
{
    return getSystemById_impl(mIdState.mGizmoSystemIdToGlobalIndex, &mAllocators.mGizmoSystemAllocator, systemId);
}

template <> FreeLookCameraSystem *World::getSystemById<FreeLookCameraSystem>(const Id &systemId) const
{
    return getSystemById_impl(mIdState.mFreeLookCameraSystemIdToGlobalIndex, &mAllocators.mFreeLookCameraSystemAllocator,
        systemId);
}

template <> TerrainSystem *World::getSystemById<TerrainSystem>(const Id &systemId) const
{
    return getSystemById_impl(mIdState.mTerrainSystemIdToGlobalIndex, &mAllocators.mTerrainSystemAllocator,
        systemId);
}

template <> AssetLoadingSystem* World::getSystemById<AssetLoadingSystem>(const Id& systemId) const
{
    return getSystemById_impl(mIdState.mAssetLoadingSystemIdToGlobalIndex, &mAllocators.mAssetLoadingSystemAllocator,
        systemId);
}





template <> RenderSystem *World::getSystemByGuid<RenderSystem>(const Guid &systemGuid) const
{
    return getSystemByGuid_impl(mIdState.mRenderSystemGuidToGlobalIndex, &mAllocators.mRenderSystemAllocator,
                                systemGuid);
}

template <> PhysicsSystem *World::getSystemByGuid<PhysicsSystem>(const Guid &systemGuid) const
{
    return getSystemByGuid_impl(mIdState.mPhysicsSystemGuidToGlobalIndex, &mAllocators.mPhysicsSystemAllocator,
                              systemGuid);
}
template <> CleanUpSystem *World::getSystemByGuid<CleanUpSystem>(const Guid &systemGuid) const
{
    return getSystemByGuid_impl(mIdState.mCleanupSystemGuidToGlobalIndex, &mAllocators.mCleanupSystemAllocator,
                              systemGuid);
}

template <> DebugSystem *World::getSystemByGuid<DebugSystem>(const Guid &systemGuid) const
{
    return getSystemByGuid_impl(mIdState.mDebugSystemGuidToGlobalIndex, &mAllocators.mDebugSystemAllocator, systemGuid);
}

template <> GizmoSystem *World::getSystemByGuid<GizmoSystem>(const Guid &systemGuid) const
{
    return getSystemByGuid_impl(mIdState.mGizmoSystemGuidToGlobalIndex, &mAllocators.mGizmoSystemAllocator, systemGuid);
}

template <> FreeLookCameraSystem *World::getSystemByGuid<FreeLookCameraSystem>(const Guid &systemGuid) const
{
    return getSystemByGuid_impl(mIdState.mFreeLookCameraSystemGuidToGlobalIndex,
                              &mAllocators.mFreeLookCameraSystemAllocator, systemGuid);
}

template <> TerrainSystem *World::getSystemByGuid<TerrainSystem>(const Guid &systemGuid) const
{
    return getSystemByGuid_impl(mIdState.mTerrainSystemGuidToGlobalIndex, &mAllocators.mTerrainSystemAllocator,
                              systemGuid);
}

template <> AssetLoadingSystem* World::getSystemByGuid<AssetLoadingSystem>(const Guid& systemGuid) const
{
    return getSystemByGuid_impl(mIdState.mAssetLoadingSystemGuidToGlobalIndex, &mAllocators.mAssetLoadingSystemAllocator,
        systemGuid);
}



template <> Mesh* World::getAssetByIndex<Mesh>(size_t index) const
{
    return mAllocators.mMeshAllocator.get(index);
}

template <> Material* World::getAssetByIndex<Material>(size_t index) const
{
    return mAllocators.mMaterialAllocator.get(index);
}

template <> Shader* World::getAssetByIndex<Shader>(size_t index) const
{
    return mAllocators.mShaderAllocator.get(index);
}

template <> Texture2D* World::getAssetByIndex<Texture2D>(size_t index) const
{
    return mAllocators.mTexture2DAllocator.get(index);
}

template <> Cubemap* World::getAssetByIndex<Cubemap>(size_t index) const
{
    return mAllocators.mCubemapAllocator.get(index);
}

template <> RenderTexture* World::getAssetByIndex<RenderTexture>(size_t index) const
{
    return mAllocators.mRenderTextureAllocator.get(index);
}

template <> Font* World::getAssetByIndex<Font>(size_t index) const
{
    return mAllocators.mFontAllocator.get(index);
}

template <> Sprite* World::getAssetByIndex<Sprite>(size_t index) const
{
    return mAllocators.mSpriteAllocator.get(index);
}

template <> Mesh* World::getAssetById<Mesh>(const Id& assetId) const
{
    return getAssetById_impl(mIdState.mMeshIdToGlobalIndex, &mAllocators.mMeshAllocator, assetId);
}

template <> Material *World::getAssetById<Material>(const Id &assetId) const
{
    return getAssetById_impl(mIdState.mMaterialIdToGlobalIndex, &mAllocators.mMaterialAllocator, assetId);
}

template <> Shader *World::getAssetById<Shader>(const Id &assetId) const
{
    return getAssetById_impl(mIdState.mShaderIdToGlobalIndex, &mAllocators.mShaderAllocator, assetId);
}

template <> Texture2D *World::getAssetById<Texture2D>(const Id &assetId) const
{
    return getAssetById_impl(mIdState.mTexture2DIdToGlobalIndex, &mAllocators.mTexture2DAllocator, assetId);
}

template <> Cubemap *World::getAssetById<Cubemap>(const Id &assetId) const
{
    return getAssetById_impl(mIdState.mCubemapIdToGlobalIndex, &mAllocators.mCubemapAllocator, assetId);
}

template <> RenderTexture *World::getAssetById<RenderTexture>(const Id &assetId) const
{
    return getAssetById_impl(mIdState.mRenderTextureIdToGlobalIndex, &mAllocators.mRenderTextureAllocator, assetId);
}

template <> Font *World::getAssetById<Font>(const Id &assetId) const
{
    return getAssetById_impl(mIdState.mFontIdToGlobalIndex, &mAllocators.mFontAllocator, assetId);
}

template <> Sprite *World::getAssetById<Sprite>(const Id &assetId) const
{
    return getAssetById_impl(mIdState.mSpriteIdToGlobalIndex, &mAllocators.mSpriteAllocator, assetId);
}





template <> Mesh *World::getAssetByGuid<Mesh>(const Guid &assetGuid) const
{
    return getAssetByGuid_impl(mIdState.mMeshGuidToGlobalIndex, &mAllocators.mMeshAllocator, assetGuid);
}

template <> Material *World::getAssetByGuid<Material>(const Guid &assetGuid) const
{
    return getAssetByGuid_impl(mIdState.mMaterialGuidToGlobalIndex, &mAllocators.mMaterialAllocator, assetGuid);
}

template <> Shader *World::getAssetByGuid<Shader>(const Guid &assetGuid) const
{
    return getAssetByGuid_impl(mIdState.mShaderGuidToGlobalIndex, &mAllocators.mShaderAllocator, assetGuid);
}

template <> Texture2D *World::getAssetByGuid<Texture2D>(const Guid &assetGuid) const
{
    return getAssetByGuid_impl(mIdState.mTexture2DGuidToGlobalIndex, &mAllocators.mTexture2DAllocator, assetGuid);
}

template <> Cubemap *World::getAssetByGuid<Cubemap>(const Guid &assetGuid) const
{
    return getAssetByGuid_impl(mIdState.mCubemapGuidToGlobalIndex, &mAllocators.mCubemapAllocator, assetGuid);
}

template <> RenderTexture *World::getAssetByGuid<RenderTexture>(const Guid &assetGuid) const
{
    return getAssetByGuid_impl(mIdState.mRenderTextureGuidToGlobalIndex, &mAllocators.mRenderTextureAllocator,
                               assetGuid);
}

template <> Font *World::getAssetByGuid<Font>(const Guid &assetGuid) const
{
    return getAssetByGuid_impl(mIdState.mFontGuidToGlobalIndex, &mAllocators.mFontAllocator, assetGuid);
}

template <> Sprite *World::getAssetByGuid<Sprite>(const Guid &assetGuid) const
{
    return getAssetByGuid_impl(mIdState.mSpriteGuidToGlobalIndex, &mAllocators.mSpriteAllocator, assetGuid);
}









template <> Mesh* World::createAsset<Mesh>()
{
    return createAsset_impl(&mAllocators.mMeshAllocator, Guid::newGuid());
}

template <> Material* World::createAsset<Material>()
{
    return createAsset_impl(&mAllocators.mMaterialAllocator, Guid::newGuid());
}

template <> Shader* World::createAsset<Shader>()
{
    return createAsset_impl(&mAllocators.mShaderAllocator, Guid::newGuid());
}

template <> Texture2D* World::createAsset<Texture2D>()
{
    return createAsset_impl(&mAllocators.mTexture2DAllocator, Guid::newGuid());
}

template <> Cubemap* World::createAsset<Cubemap>()
{
    return createAsset_impl(&mAllocators.mCubemapAllocator, Guid::newGuid());
}

template <> RenderTexture* World::createAsset<RenderTexture>()
{
    return createAsset_impl(&mAllocators.mRenderTextureAllocator, Guid::newGuid());
}

template <> Font* World::createAsset<Font>()
{
    return createAsset_impl(&mAllocators.mFontAllocator, Guid::newGuid());
}

template <> Sprite* World::createAsset<Sprite>()
{
    return createAsset_impl(&mAllocators.mSpriteAllocator, Guid::newGuid());
}

template <> Mesh* World::createAsset<Mesh>(const Guid& assetGuid)
{
    return createAsset_impl(&mAllocators.mMeshAllocator, assetGuid);
}

template <> Material *World::createAsset<Material>(const Guid &assetGuid)
{
    return createAsset_impl(&mAllocators.mMaterialAllocator, assetGuid);
}

template <> Shader *World::createAsset<Shader>(const Guid &assetGuid)
{
    return createAsset_impl(&mAllocators.mShaderAllocator, assetGuid);
}

template <> Texture2D *World::createAsset<Texture2D>(const Guid &assetGuid)
{
    return createAsset_impl(&mAllocators.mTexture2DAllocator, assetGuid);
}

template <> Cubemap *World::createAsset<Cubemap>(const Guid &assetGuid)
{
    return createAsset_impl(&mAllocators.mCubemapAllocator, assetGuid);
}

template <> RenderTexture *World::createAsset<RenderTexture>(const Guid &assetGuid)
{
    return createAsset_impl(&mAllocators.mRenderTextureAllocator, assetGuid);
}

template <> Font *World::createAsset<Font>(const Guid &assetGuid)
{
    return createAsset_impl(&mAllocators.mFontAllocator, assetGuid);
}

template <> Sprite *World::createAsset<Sprite>(const Guid &assetGuid)
{
    return createAsset_impl(&mAllocators.mSpriteAllocator, assetGuid);
}

template <> Mesh* World::createAsset<Mesh>(const YAML::Node& in)
{
    return createAsset_impl(&mAllocators.mMeshAllocator, in);
}

template <> Material* World::createAsset<Material>(const YAML::Node& in)
{
    return createAsset_impl(&mAllocators.mMaterialAllocator, in);
}

template <> Shader* World::createAsset<Shader>(const YAML::Node& in)
{
    return createAsset_impl(&mAllocators.mShaderAllocator, in);
}

template <> Texture2D* World::createAsset<Texture2D>(const YAML::Node& in)
{
    return createAsset_impl(&mAllocators.mTexture2DAllocator, in);
}

template <> Cubemap* World::createAsset<Cubemap>(const YAML::Node& in)
{
    return createAsset_impl(&mAllocators.mCubemapAllocator, in);
}

template <> RenderTexture* World::createAsset<RenderTexture>(const YAML::Node& in)
{
    return createAsset_impl(&mAllocators.mRenderTextureAllocator, in);
}

template <> Font* World::createAsset<Font>(const YAML::Node& in)
{
    return createAsset_impl(&mAllocators.mFontAllocator, in);
}

template <> Sprite* World::createAsset<Sprite>(const YAML::Node& in)
{
    return createAsset_impl(&mAllocators.mSpriteAllocator, in);
}

void World::addToIdState(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mGuidToGlobalIndex[guid] = index;
    mIdState.mIdToGlobalIndex[id] = index;

    mIdState.mGuidToType[guid] = type;
    mIdState.mIdToType[id] = type;

    mIdState.mGuidToId[guid] = id;
    mIdState.mIdToGuid[id] = guid;
}

void World::removeFromIdState(const Guid &guid, const Id &id)
{
    mIdState.mGuidToGlobalIndex.erase(guid);
    mIdState.mIdToGlobalIndex.erase(id);

    mIdState.mGuidToType.erase(guid);
    mIdState.mIdToType.erase(id);

    mIdState.mGuidToId.erase(guid);
    mIdState.mIdToGuid.erase(id);
}

template <> void World::addToIdState_impl<Scene>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mSceneGuidToGlobalIndex[guid] = index;
    mIdState.mSceneIdToGlobalIndex[id] = index;
    
    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<Mesh>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mMeshGuidToGlobalIndex[guid] = index;
    mIdState.mMeshIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<Material>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mMaterialGuidToGlobalIndex[guid] = index;
    mIdState.mMaterialIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<Shader>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mShaderGuidToGlobalIndex[guid] = index;
    mIdState.mShaderIdToGlobalIndex[id] = index;
    
    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<Texture2D>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mTexture2DGuidToGlobalIndex[guid] = index;
    mIdState.mTexture2DIdToGlobalIndex[id] = index;
    
    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<Cubemap>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mCubemapGuidToGlobalIndex[guid] = index;
    mIdState.mCubemapIdToGlobalIndex[id] = index;
    
    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<RenderTexture>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mRenderTextureGuidToGlobalIndex[guid] = index;
    mIdState.mRenderTextureIdToGlobalIndex[id] = index;
    
    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<Font>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mFontGuidToGlobalIndex[guid] = index;
    mIdState.mFontIdToGlobalIndex[id] = index;
    
    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<Sprite>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mSpriteGuidToGlobalIndex[guid] = index;
    mIdState.mSpriteIdToGlobalIndex[id] = index;
    
    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<RenderSystem>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mRenderSystemGuidToGlobalIndex[guid] = index;
    mIdState.mRenderSystemIdToGlobalIndex[id] = index;
    
    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<PhysicsSystem>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mPhysicsSystemGuidToGlobalIndex[guid] = index;
    mIdState.mPhysicsSystemIdToGlobalIndex[id] = index;
    
    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<CleanUpSystem>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mCleanupSystemGuidToGlobalIndex[guid] = index;
    mIdState.mCleanupSystemIdToGlobalIndex[id] = index;
    
    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<DebugSystem>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mDebugSystemGuidToGlobalIndex[guid] = index;
    mIdState.mDebugSystemIdToGlobalIndex[id] = index;
    
    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<GizmoSystem>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mGizmoSystemGuidToGlobalIndex[guid] = index;
    mIdState.mGizmoSystemIdToGlobalIndex[id] = index;
    
    addToIdState(guid, id, index, type);
}

template <>
void World::addToIdState_impl<FreeLookCameraSystem>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mFreeLookCameraSystemGuidToGlobalIndex[guid] = index;
    mIdState.mFreeLookCameraSystemIdToGlobalIndex[id] = index;
    
    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<TerrainSystem>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mTerrainSystemGuidToGlobalIndex[guid] = index;
    mIdState.mTerrainSystemIdToGlobalIndex[id] = index;
    
    addToIdState(guid, id, index, type);
}

template <> void World::addToIdState_impl<AssetLoadingSystem>(const Guid& guid, const Id& id, int index, int type)
{
    mIdState.mAssetLoadingSystemGuidToGlobalIndex[guid] = index;
    mIdState.mAssetLoadingSystemIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void World::removeFromIdState_impl<Scene>(const Guid &guid, const Id &id)
{
    mIdState.mSceneGuidToGlobalIndex.erase(guid);
    mIdState.mSceneIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<Mesh>(const Guid &guid, const Id &id)
{
    mIdState.mMeshGuidToGlobalIndex.erase(guid);
    mIdState.mMeshIdToGlobalIndex.erase(id);
   
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<Material>(const Guid &guid, const Id &id)
{
    mIdState.mMaterialGuidToGlobalIndex.erase(guid);
    mIdState.mMaterialIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<Shader>(const Guid &guid, const Id &id)
{
    mIdState.mShaderGuidToGlobalIndex.erase(guid);
    mIdState.mShaderIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<Texture2D>(const Guid &guid, const Id &id)
{
    mIdState.mTexture2DGuidToGlobalIndex.erase(guid);
    mIdState.mTexture2DIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<Cubemap>(const Guid &guid, const Id &id)
{
    mIdState.mCubemapGuidToGlobalIndex.erase(guid);
    mIdState.mCubemapIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<RenderTexture>(const Guid &guid, const Id &id)
{
    mIdState.mRenderTextureGuidToGlobalIndex.erase(guid);
    mIdState.mRenderTextureIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<Font>(const Guid &guid, const Id &id)
{
    mIdState.mFontGuidToGlobalIndex.erase(guid);
    mIdState.mFontIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<Sprite>(const Guid &guid, const Id &id)
{
    mIdState.mSpriteGuidToGlobalIndex.erase(guid);
    mIdState.mSpriteIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<RenderSystem>(const Guid &guid, const Id &id)
{
    mIdState.mRenderSystemGuidToGlobalIndex.erase(guid);
    mIdState.mRenderSystemIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<PhysicsSystem>(const Guid &guid, const Id &id)
{
    mIdState.mPhysicsSystemGuidToGlobalIndex.erase(guid);
    mIdState.mPhysicsSystemIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<CleanUpSystem>(const Guid &guid, const Id &id)
{
    mIdState.mCleanupSystemGuidToGlobalIndex.erase(guid);
    mIdState.mCleanupSystemIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<DebugSystem>(const Guid &guid, const Id &id)
{
    mIdState.mDebugSystemGuidToGlobalIndex.erase(guid);
    mIdState.mDebugSystemIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<GizmoSystem>(const Guid &guid, const Id &id)
{
    mIdState.mGizmoSystemGuidToGlobalIndex.erase(guid);
    mIdState.mGizmoSystemIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<FreeLookCameraSystem>(const Guid &guid, const Id &id)
{
    mIdState.mFreeLookCameraSystemGuidToGlobalIndex.erase(guid);
    mIdState.mFreeLookCameraSystemIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<TerrainSystem>(const Guid &guid, const Id &id)
{
    mIdState.mTerrainSystemGuidToGlobalIndex.erase(guid);
    mIdState.mTerrainSystemIdToGlobalIndex.erase(id);
    
    removeFromIdState(guid, id);
}

template <> void World::removeFromIdState_impl<AssetLoadingSystem>(const Guid& guid, const Id& id)
{
    mIdState.mAssetLoadingSystemGuidToGlobalIndex.erase(guid);
    mIdState.mAssetLoadingSystemIdToGlobalIndex.erase(id);

    removeFromIdState(guid, id);
}

template <typename T> T* World::addSystem_impl(PoolAllocator<T>* allocator, size_t order)
{
    static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

    int systemGlobalIndex = (int)allocator->getCount();
    int systemType = SystemType<T>::type;

    T* system = allocator->construct(this, Guid::newGuid(), Id::newId());

    if (system != nullptr)
    {
        addToIdState_impl<T>(system->getGuid(), system->getId(), systemGlobalIndex, systemType);

        size_t locationToInsert = mSystems.size();
        for (size_t i = 0; i < mSystems.size(); i++)
        {
            if (order < mSystems[i]->getOrder())
            {
                locationToInsert = i;
                break;
            }
        }

        mSystems.insert(mSystems.begin() + locationToInsert, system);
    }

    return system;
}

template <typename T>
T* World::getSystemById_impl(const std::unordered_map<Id, int>& idToIndexMap, const PoolAllocator<T>* allocator,
                             const Id &systemId) const
{
    static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

    if (allocator == nullptr || SystemType<T>::type != getTypeOf(systemId))
    {
        return nullptr;
    }

    return getById_impl<T>(idToIndexMap, allocator, systemId);
}

template <typename T>
T *World::getSystemByGuid_impl(const std::unordered_map<Guid, int> &guidToIndexMap, const PoolAllocator<T> *allocator,
                             const Guid &systemGuid) const
{
    static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

    if (allocator == nullptr || SystemType<T>::type != getTypeOf(systemGuid))
    {
        return nullptr;
    }

    return getByGuid_impl<T>(guidToIndexMap, allocator, systemGuid);
}

template <typename T>
T* World::getAssetById_impl(const std::unordered_map<Id, int>& idToIndexMap, const PoolAllocator<T>* allocator,
    const Id& assetId) const
{
    static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

    if (allocator == nullptr || AssetType<T>::type != getTypeOf(assetId))
    {
        return nullptr;
    }

    return getById_impl<T>(idToIndexMap, allocator, assetId);
}

template <typename T>
T *World::getAssetByGuid_impl(const std::unordered_map<Guid, int> &guidToIndexMap, const PoolAllocator<T> *allocator,
                            const Guid &assetGuid) const
{
    static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

    if (allocator == nullptr || AssetType<T>::type != getTypeOf(assetGuid))
    {
        return nullptr;
    }

    return getByGuid_impl<T>(guidToIndexMap, allocator, assetGuid);
}

template <typename T> T* World::createAsset_impl(PoolAllocator<T>* allocator, const Guid& assetGuid)
{
    static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

    int index = (int)allocator->getCount();
    int type = AssetType<T>::type;

    T* asset = allocator->construct(this, assetGuid, Id::newId());

    if (asset != nullptr)
    {
        addToIdState_impl<T>(asset->getGuid(), asset->getId(), index, type);
    }

    return asset;
}

template <typename T> T* World::createAsset_impl(PoolAllocator<T>* allocator, const YAML::Node& in)
{
    static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

    int index = (int)allocator->getCount();
    int type = AssetType<T>::type;

    T* asset = allocator->construct(this, in, Id::newId());

    if (asset != nullptr)
    {
        addToIdState_impl<T>(asset->getGuid(), asset->getId(), index, type);
    }

    return asset;
}

template <typename T>
T* World::getById_impl(const std::unordered_map<Id, int>& idToIndexMap, const PoolAllocator<T>* allocator,
    const Id& id) const
{
    static_assert(std::is_base_of<Scene, T>() || std::is_base_of<Asset, T>() || std::is_base_of<System, T>(),
        "'T' is not of type Asset or System");

    std::unordered_map<Id, int>::const_iterator it = idToIndexMap.find(id);
    if (it != idToIndexMap.end())
    {
        return allocator->get(it->second);
    }
    else
    {
        return nullptr;
    }
}

template <typename T>
T *World::getByGuid_impl(const std::unordered_map<Guid, int> &guidToIndexMap, const PoolAllocator<T> *allocator,
                       const Guid &guid) const
{
    static_assert(std::is_base_of<Scene, T>() || std::is_base_of<Asset, T>() || std::is_base_of<System, T>(),
                  "'T' is not of type Asset or System");

    std::unordered_map<Guid, int>::const_iterator it = guidToIndexMap.find(guid);
    if (it != guidToIndexMap.end())
    {
        return allocator->get(it->second);
    }
    else
    {
        return nullptr;
    }
}


World::World()
{
    mActiveScene = createScene();
    mPrimitives.createPrimitiveMeshes(this, 10, 10);
}

World::~World()
{
}

void World::loadAssetsInPath(const std::filesystem::path &filePath)
{
    if (std::filesystem::is_directory(filePath))
    {
        std::stack<std::filesystem::path> stack;
        stack.push(filePath);

        while (!stack.empty())
        {
            std::filesystem::path currentPath = stack.top();
            stack.pop();

            std::error_code error_code;
            for (const std::filesystem::directory_entry &entry :
                 std::filesystem::directory_iterator(currentPath, error_code))
            {
                if (std::filesystem::is_directory(entry, error_code))
                {
                    stack.push(entry.path());
                }
                else if (std::filesystem::is_regular_file(entry, error_code))
                {
                    std::string extension = entry.path().extension().string();
                    if (extension == ".mesh" || extension == ".shader" || extension == ".material" ||
                        extension == ".texture")
                    {
                        std::filesystem::path relativeDataPath =
                            entry.path().lexically_relative(std::filesystem::current_path());

                        std::cout << "relative data path: " << relativeDataPath.string() << std::endl;

                        loadAssetFromYAML(relativeDataPath.string());
                    }
                }
            }
        }
    }
}

Asset *World::loadAssetFromYAML(const std::string &filePath)
{
    YAML::Node in;
    try
    {
        in = YAML::LoadFile(filePath);
    }
    catch (YAML::Exception e /*YAML::BadFile e*/)
    {
        Log::error("YAML exception hit when trying to load file");
        return nullptr;
    }

    if (!in.IsMap() || in.begin() == in.end())
    {
        return nullptr;
    }

    if (in.begin()->first.IsScalar() && in.begin()->second.IsMap())
    {
        int type = YAML::getValue<int>(in.begin()->second, "type");
        Guid guid = YAML::getValue<Guid>(in.begin()->second, "id");

        if (PhysicsEngine::isAsset(type) && guid.isValid())
        {
            generateSourcePaths(filePath, in.begin()->second);

            Asset *asset = getAssetByGuid(guid, type);
            if (asset != nullptr)
            {
                asset->deserialize(in);
            }

            if (asset == nullptr)
            {
                asset = createAsset(in.begin()->second, type);  
            }

            if (asset != nullptr)
            {
                mIdState.mAssetGuidToFilepath[asset->getGuid()] = filePath;
                mIdState.mAssetFilepathToGuid[filePath] = asset->getGuid();
            }

            return asset;
        }
    }

    return nullptr;
}

Scene *World::loadSceneFromYAML(const std::string &filePath)
{
    YAML::Node in;
    try
    {
        in = YAML::LoadFile(filePath);
    }
    catch (YAML::BadFile e)
    {
        Log::error("YAML exception hit when trying to load file");
        return nullptr;
    }

    int type = YAML::getValue<int>(in, "type");
    Guid guid = YAML::getValue<Guid>(in, "id");

    if (PhysicsEngine::isScene(type) && guid.isValid())
    {
        Scene *scene = getSceneByGuid(guid);
        if (scene != nullptr)
        {
            scene->deserialize(in);
        }
        else
        {
            scene = createScene(in);
        }

        if (scene != nullptr)
        {
            mIdState.mSceneGuidToFilepath[scene->getGuid()] = filePath;
            mIdState.mSceneFilepathToGuid[filePath] = scene->getGuid();

            // Copy 'do not destroy' entities from old scene to new scene
            copyDoNotDestroyEntities(mActiveScene, scene);

            mActiveScene = scene;
        }

        return scene;
    }

    return nullptr;
}

bool World::writeAssetToYAML(const std::string &filePath, const Guid &assetGuid) const
{
    int type = getTypeOf(assetGuid);

    Asset *asset = getAssetByGuid(assetGuid, type);
    if (asset == nullptr)
    {
        return false;    
    }

    return asset->writeToYAML(filePath);
}

bool World::writeSceneToYAML(const std::string &filePath, const Guid &sceneGuid) const
{
    Scene *scene = getSceneByGuid(sceneGuid);
    if (scene == nullptr)
    {
        return false;
    }

    return scene->writeToYAML(filePath);
}

void World::copyDoNotDestroyEntities(Scene *from, Scene *to)
{
    for (size_t i = 0; i < from->getNumberOfEntities(); i++)
    {
        Entity *entity = from->getEntityByIndex(i);
        if (entity->mDoNotDestroy)
        {
            YAML::Node entityNode;
            entity->serialize(entityNode);

            std::cout << "do not destroy entity: " << entity->getGuid().toString() << std::endl;

            Entity *newEntity = to->getEntityByGuid(entity->getGuid());
            if (newEntity != nullptr)
            {
                newEntity->deserialize(entityNode);
            }
            else
            {
                newEntity = to->createEntity(entityNode);   
            }

            std::vector<std::pair<Guid, int>> components = entity->getComponentsOnEntity();
            for (size_t j = 0; j < components.size(); j++)
            {
                Component *component = from->getComponentByGuid(components[j].first, components[j].second);

                YAML::Node componentNode;
                component->serialize(componentNode);

                Component *newComponent = to->getComponentByGuid(components[j].first, components[j].second);
                if (newComponent != nullptr)
                {
                    newComponent->deserialize(componentNode);
                }
                else
                {
                    to->addComponent(componentNode, component->getType());
                }
            }
        }
    }
}

void World::generateSourcePaths(const std::string &filepath, YAML::Node &in)
{
    int type = YAML::getValue<int>(in, "type");

    std::filesystem::path path = filepath;
    path.remove_filename();

    if (PhysicsEngine::isAsset(type))
    {
        switch (type)
        {
        case AssetType<Shader>::type:
        case AssetType<Texture2D>::type:
        case AssetType<Mesh>::type: {
            std::filesystem::path source = YAML::getValue<std::string>(in, "source");
            in["sourceFilepath"] = (path / source).string();
            break;
        }
        }
    }
}

std::vector<ShaderUniform> World::getCachedMaterialUniforms(const Guid &materialGuid, const Guid &shaderGuid)
{
    return mMaterialUniformCache[materialGuid][shaderGuid];
}

void World::cacheMaterialUniforms(const Guid &materialGuid, const Guid &shaderGuid, const std::vector<ShaderUniform> &uniforms)
{
    assert(materialGuid != Guid::INVALID);
    assert(shaderGuid != Guid::INVALID);

    mMaterialUniformCache[materialGuid][shaderGuid] = uniforms;
}

size_t World::getNumberOfScenes() const
{
    return mAllocators.mSceneAllocator.getCount();
}

size_t World::getNumberOfUpdatingSystems() const
{
    return mSystems.size();
}

Mesh *World::getPrimtiveMesh(PrimitiveType type) const
{
    switch (type)
    {
    case PrimitiveType::Plane:
        return getAssetByGuid<Mesh>(mPrimitives.mPlaneMeshGuid);
    case PrimitiveType::Disc:
        return getAssetByGuid<Mesh>(mPrimitives.mDiscMeshGuid);
    case PrimitiveType::Cube:
        return getAssetByGuid<Mesh>(mPrimitives.mCubeMeshGuid);
    case PrimitiveType::Sphere:
        return getAssetByGuid<Mesh>(mPrimitives.mSphereMeshGuid);
    case PrimitiveType::Cylinder:
        return getAssetByGuid<Mesh>(mPrimitives.mCylinderMeshGuid);
    case PrimitiveType::Cone:
        return getAssetByGuid<Mesh>(mPrimitives.mConeMeshGuid);
    default:
        return nullptr;
    }
}


Material *World::getPrimtiveMaterial() const
{
    return getAssetByGuid<Material>(mPrimitives.mStandardMaterialGuid);
}

Asset *World::getAssetByGuid(const Guid &assetGuid, int type) const
{
    switch (type)
    {
    case AssetType<Mesh>::type: {
        return getAssetByGuid<Mesh>(assetGuid);
    }
    case AssetType<Material>::type: {
        return getAssetByGuid<Material>(assetGuid);
    }
    case AssetType<Shader>::type: {
        return getAssetByGuid<Shader>(assetGuid);
    }
    case AssetType<Texture2D>::type: {
        return getAssetByGuid<Texture2D>(assetGuid);
    }
    case AssetType<Cubemap>::type: {
        return getAssetByGuid<Cubemap>(assetGuid);
    }
    case AssetType<RenderTexture>::type: {
        return getAssetByGuid<RenderTexture>(assetGuid);
    }
    case AssetType<Sprite>::type: {
        return getAssetByGuid<Sprite>(assetGuid);
    }
    case AssetType<Font>::type: {
        return getAssetByGuid<Font>(assetGuid);
    }
    }

    return nullptr;
}

Asset *World::getAssetById(const Id &assetId, int type) const
{
    switch (type)
    {
    case AssetType<Mesh>::type: {return getAssetById<Mesh>(assetId);}
    case AssetType<Material>::type: {return getAssetById<Material>(assetId);}
    case AssetType<Shader>::type: {return getAssetById<Shader>(assetId);}
    case AssetType<Texture2D>::type: {return getAssetById<Texture2D>(assetId);}
    case AssetType<Cubemap>::type: {return getAssetById<Cubemap>(assetId);}
    case AssetType<RenderTexture>::type: {return getAssetById<RenderTexture>(assetId);}
    case AssetType<Sprite>::type: {return getAssetById<Sprite>(assetId);}
    case AssetType<Font>::type: {return getAssetById<Font>(assetId);}
    }

    return nullptr;
}

Scene *World::getSceneById(const Id &sceneId) const
{
    return getById_impl<Scene>(mIdState.mSceneIdToGlobalIndex, &mAllocators.mSceneAllocator, sceneId);
}

Scene *World::getSceneByGuid(const Guid &sceneGuid) const
{
    return getByGuid_impl<Scene>(mIdState.mSceneGuidToGlobalIndex, &mAllocators.mSceneAllocator, sceneGuid);
}

Scene *World::getSceneByIndex(size_t index) const
{
    return mAllocators.mSceneAllocator.get(index);
}

System *World::getSystemByUpdateOrder(size_t order) const
{
    if (order >= mSystems.size())
    {
        return nullptr;
    }

    return mSystems[order];
}

int World::getIndexOf(const Id &id) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mIdToGlobalIndex.find(id);
    if (it != mIdState.mIdToGlobalIndex.end())
    {
        return it->second;
    }

    return -1;
}

int World::getIndexOf(const Guid &guid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mGuidToGlobalIndex.find(guid);
    if (it != mIdState.mGuidToGlobalIndex.end())
    {
        return it->second;
    }

    return -1;
}

int World::getTypeOf(const Id &id) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mIdToType.find(id);
    if (it != mIdState.mIdToType.end())
    {
        return it->second;
    }

    return -1;
}

int World::getTypeOf(const Guid &guid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mGuidToType.find(guid);
    if (it != mIdState.mGuidToType.end())
    {
        return it->second;
    }

    return -1;
}

Scene *World::createScene()
{
    int globalIndex = (int)mAllocators.mSceneAllocator.getCount();
    int type = SceneType<Scene>::type;

    Scene *scene = mAllocators.mSceneAllocator.construct(this, Guid::newGuid(), Id::newId());

    if (scene != nullptr)
    {
        addToIdState_impl<Scene>(scene->getGuid(), scene->getId(), globalIndex, type);
    }

    return scene;
}

Scene *World::createScene(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mSceneAllocator.getCount();
    int type = SceneType<Scene>::type;

    Scene *scene = mAllocators.mSceneAllocator.construct(this, in, Id::newId());

    if (scene != nullptr)
    {
        addToIdState_impl<Scene>(scene->getGuid(), scene->getId(), globalIndex, type);
    }

    return scene;
}

Asset *World::createAsset(const YAML::Node &in, int type)
{
    switch (type)
    {
    case AssetType<Mesh>::type: {
        return createAsset<Mesh>(in);
    }
    case AssetType<Material>::type: {
        return createAsset<Material>(in);
    }
    case AssetType<Shader>::type: {
        return createAsset<Shader>(in);
    }
    case AssetType<Texture2D>::type: {
        return createAsset<Texture2D>(in);
    }
    case AssetType<Cubemap>::type: {
        return createAsset<Cubemap>(in);
    }
    case AssetType<RenderTexture>::type: {
        return createAsset<RenderTexture>(in);
    }
    case AssetType<Sprite>::type: {
        return createAsset<Sprite>(in);
    }
    case AssetType<Font>::type: {
        return createAsset<Font>(in);
    }
    }

    return nullptr;
}

void World::latentDestroyAsset(const Guid &assetGuid, int assetType)
{
    mIdState.mAssetGuidsMarkedLatentDestroy.push_back(std::make_pair(assetGuid, assetType));
}

void World::immediateDestroyAsset(const Guid &assetGuid, int assetType)
{
    int index = getIndexOf(assetGuid);

    Id assetId = mIdState.mGuidToId[assetGuid];
    
    if (assetType == AssetType<Material>::type)
    {
        Asset *swap = mAllocators.mMaterialAllocator.destruct(index);
    
        removeFromIdState_impl<Material>(assetGuid, assetId);

        if (swap != nullptr)
        {
            addToIdState_impl<Material>(swap->getGuid(), swap->getId(), index, assetType);
        }
    }
    else if (assetType == AssetType<Mesh>::type)
    {
        Asset *swap = mAllocators.mMeshAllocator.destruct(index);
    
        removeFromIdState_impl<Mesh>(assetGuid, assetId);

        if (swap != nullptr)
        {
            addToIdState_impl<Mesh>(swap->getGuid(), swap->getId(), index, assetType);
        }
    }
    else if (assetType == AssetType<Shader>::type)
    {
        Asset *swap = mAllocators.mShaderAllocator.destruct(index);
    
        removeFromIdState_impl<Shader>(assetGuid, assetId);

        if (swap != nullptr)
        {
            addToIdState_impl<Shader>(swap->getGuid(), swap->getId(), index, assetType);
        }
    }
    else if (assetType == AssetType<Texture2D>::type)
    {
        Asset *swap = mAllocators.mTexture2DAllocator.destruct(index);
    
        removeFromIdState_impl<Texture2D>(assetGuid, assetId);

        if (swap != nullptr)
        {
            addToIdState_impl<Texture2D>(swap->getGuid(), swap->getId(), index, assetType);
        }
    }
    else if (assetType == AssetType<Cubemap>::type)
    {
        Asset *swap = mAllocators.mCubemapAllocator.destruct(index);
    
        removeFromIdState_impl<Cubemap>(assetGuid, assetId);

        if (swap != nullptr)
        {
            addToIdState_impl<Cubemap>(swap->getGuid(), swap->getId(), index, assetType);
        }
    }
    else if (assetType == AssetType<RenderTexture>::type)
    {
        Asset *swap = mAllocators.mRenderTextureAllocator.destruct(index);
    
        removeFromIdState_impl<RenderTexture>(assetGuid, assetId);

        if (swap != nullptr)
        {
            addToIdState_impl<RenderTexture>(swap->getGuid(), swap->getId(), index, assetType);
        }
    }
    else if (assetType == AssetType<Font>::type)
    {
        Asset *swap = mAllocators.mFontAllocator.destruct(index);
    
        removeFromIdState_impl<Font>(assetGuid, assetId);

        if (swap != nullptr)
        {
            addToIdState_impl<Font>(swap->getGuid(), swap->getId(), index, assetType);
        }
    }
    else if (assetType == AssetType<Sprite>::type)
    {
        Asset *swap = mAllocators.mSpriteAllocator.destruct(index);
    
        removeFromIdState_impl<Sprite>(assetGuid, assetId);

        if (swap != nullptr)
        {
            addToIdState_impl<Sprite>(swap->getGuid(), swap->getId(), index, assetType);
        }
    }
    else
    {
        std::string message = "Error: Invalid asset instance type (" + std::to_string(assetType) +
                                ") when trying to destroy internal asset\n";
        Log::error(message.c_str());
    }
}

std::string World::getAssetFilepath(const Guid &assetGuid) const
{
    std::unordered_map<Guid, std::string>::const_iterator it = mIdState.mAssetGuidToFilepath.find(assetGuid);
    if (it != mIdState.mAssetGuidToFilepath.end())
    {
        return it->second;
    }

    return std::string();
}

std::string World::getSceneFilepath(const Guid &sceneGuid) const
{
    std::unordered_map<Guid, std::string>::const_iterator it = mIdState.mSceneGuidToFilepath.find(sceneGuid);
    if (it != mIdState.mSceneGuidToFilepath.end())
    {
        return it->second;
    }

    return std::string();
}

Guid World::getAssetGuid(const std::string& filepath) const
{
    std::unordered_map<std::string, Guid>::const_iterator it = mIdState.mAssetFilepathToGuid.find(filepath);
    if (it != mIdState.mAssetFilepathToGuid.end())
    {
        return it->second;
    }

    return Guid::INVALID;
}

Guid World::getSceneGuid(const std::string& filepath) const
{
    std::unordered_map<std::string, Guid>::const_iterator it = mIdState.mSceneFilepathToGuid.find(filepath);
    if (it != mIdState.mSceneFilepathToGuid.end())
    {
        return it->second;
    }

    return Guid::INVALID;
}

Scene *World::getActiveScene()
{
    return mActiveScene;
}

// bool World::raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance)
//{
//	Ray ray;
//
//	ray.origin = origin;
//	ray.direction = direction;
//
//	return sgrid.intersect(ray) != NULL;// || dtree.intersect(ray) != NULL;
//	// return stree.intersect(ray) != NULL || dtree.intersect(ray) != NULL;
//}
//
//// begin by only implementing for spheres first and later I will add for bounds, capsules etc
// bool World::raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance, Collider** collider)
//{
//	Ray ray;
//
//	ray.origin = origin;
//	ray.direction = direction;
//
//	// Object* object = stree.intersect(ray);
//	BoundingSphere* boundingSphere = sgrid.intersect(ray);
//
//	if(boundingSphere != NULL){
//		//std::cout << "AAAAAA id: " << boundingSphere->id.toString() << std::endl;
//		std::map<Guid, int>::iterator it = idToGlobalIndex.find(boundingSphere->id);
//		if(it != idToGlobalIndex.end()){
//			int colliderIndex = it->second;
//
//			if(boundingSphere->primitiveType == 0){
//				*collider = getComponentByIndex<SphereCollider>(colliderIndex);
//			}
//			else if(boundingSphere->primitiveType == 1){
//				*collider = getComponentByIndex<BoxCollider>(colliderIndex);
//			}
//			else{
//				*collider = getComponentByIndex<MeshCollider>(colliderIndex);
//			}
//			return true;
//		}
//		else{
//			std::cout << "Error: component id does not correspond to a global index" << std::endl;
//			return false;
//		}
//	}
//
//	return false;
//}