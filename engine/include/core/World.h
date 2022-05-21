#ifndef WORLD_H__
#define WORLD_H__

#include <assert.h>
#include <string>
#include <unordered_map>
#include <filesystem>

#define GLM_FORCE_RADIANS

#include "WorldAllocators.h"
#include "WorldIdState.h"
#include "WorldPrimitives.h"

namespace PhysicsEngine
{
class World
{
  private:
    // allocators for scenes, assets, and systems
    WorldAllocators mAllocators;

    // id state for scenes, assets, and systems
    WorldIdState mIdState;

    // Primitive meshes all worlds have access to
    WorldPrimitives mPrimitives;

    // active scene
    Scene *mActiveScene;

    // all systems in world listed in order they should be updated
    std::vector<System *> mSystems;

    std::unordered_map<Guid, std::unordered_map<Guid, std::vector<ShaderUniform>>> mMaterialUniformCache;

  public:
    std::vector<Sphere> mBoundingSpheres;

  public:
    World();
    ~World();
    World(const World &other) = delete;
    World &operator=(const World &other) = delete;

    void loadAssetsInPath(const std::filesystem::path &filePath);
    Asset *loadAssetFromYAML(const std::string &filePath);
    Scene *loadSceneFromYAML(const std::string &filePath);
    bool writeAssetToYAML(const std::string &filePath, const Guid &assetId) const;
    bool writeSceneToYAML(const std::string &filePath, const Guid &sceneId) const;

    std::vector<ShaderUniform> getCachedMaterialUniforms(const Guid &materialId, const Guid &shaderId);
    void cacheMaterialUniforms(const Guid &materialId, const Guid &shaderId, const std::vector<ShaderUniform>& uniforms);

    size_t getNumberOfScenes() const;
    size_t getNumberOfUpdatingSystems() const;
    Mesh *getPrimtiveMesh(PrimitiveType type) const;
    Material *World::getPrimtiveMaterial() const;

    Scene *getActiveScene();

    Asset *getAssetById(const Guid &assetId, int type) const;
    Scene *getSceneByIndex(size_t index) const;
    Scene *getSceneById(const Guid &sceneId) const;
    System *getSystemByUpdateOrder(size_t order) const;

    Asset *createAsset(const YAML::Node &in, int type);

    int getIndexOf(const Guid &id) const;
    int getTypeOf(const Guid &id) const;

    Scene *createScene();
    Scene *World::createScene(const YAML::Node &in);
    void latentDestroyAsset(const Guid &assetId, int assetType);
    void immediateDestroyAsset(const Guid &assetId, int assetType);

    std::string getAssetFilepath(const Guid &assetId) const;
    std::string getSceneFilepath(const Guid &sceneId) const;

    Guid getAssetId(const std::string &filepath) const;
    Guid getSceneId(const std::string &filepath) const;

    template <typename T> size_t getNumberOfSystems() const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");
        return 0;
    }

    template <> size_t getNumberOfSystems<RenderSystem>() const
    {
        return mAllocators.mRenderSystemAllocator.getCount();
    }

    template <> size_t getNumberOfSystems<PhysicsSystem>() const
    {
        return mAllocators.mPhysicsSystemAllocator.getCount();
    }
    template <> size_t getNumberOfSystems<CleanUpSystem>() const
    {
        return mAllocators.mCleanupSystemAllocator.getCount();
    }

    template <> size_t getNumberOfSystems<DebugSystem>() const
    {
        return mAllocators.mDebugSystemAllocator.getCount();
    }

    template <> size_t getNumberOfSystems<GizmoSystem>() const
    {
        return mAllocators.mGizmoSystemAllocator.getCount();
    }

    template <> size_t getNumberOfSystems<FreeLookCameraSystem>() const
    {
        return mAllocators.mFreeLookCameraSystemAllocator.getCount();
    }

    template <> size_t getNumberOfSystems<TerrainSystem>() const
    {
        return mAllocators.mTerrainSystemAllocator.getCount();
    }

    template <typename T> size_t getNumberOfAssets() const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");
        return 0;
    }

    template <> size_t getNumberOfAssets<Mesh>() const
    {
        return mAllocators.mMeshAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Material>() const
    {
        return mAllocators.mMaterialAllocator.getCount();
    }
    template <> size_t getNumberOfAssets<Shader>() const
    {
        return mAllocators.mShaderAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Texture2D>() const
    {
        return mAllocators.mTexture2DAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Texture3D>() const
    {
        return mAllocators.mTexture3DAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Cubemap>() const
    {
        return mAllocators.mCubemapAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<RenderTexture>() const
    {
        return mAllocators.mRenderTextureAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Font>() const
    {
        return mAllocators.mFontAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Sprite>() const
    {
        return mAllocators.mSpriteAllocator.getCount();
    }

    template <typename T> T *getSystem() const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return nullptr;
    }

    template <> RenderSystem *getSystem<RenderSystem>() const
    {
        return getSystem_impl(&mAllocators.mRenderSystemAllocator);
    }

    template <> PhysicsSystem *getSystem<PhysicsSystem>() const
    {
        return getSystem_impl(&mAllocators.mPhysicsSystemAllocator);
    }
    template <> CleanUpSystem *getSystem<CleanUpSystem>() const
    {
        return getSystem_impl(&mAllocators.mCleanupSystemAllocator);
    }

    template <> DebugSystem *getSystem<DebugSystem>() const
    {
        return getSystem_impl(&mAllocators.mDebugSystemAllocator);
    }

    template <> GizmoSystem *getSystem<GizmoSystem>() const
    {
        return getSystem_impl(&mAllocators.mGizmoSystemAllocator);
    }

    template <> FreeLookCameraSystem *getSystem<FreeLookCameraSystem>() const
    {
        return getSystem_impl(&mAllocators.mFreeLookCameraSystemAllocator);
    }

    template <> TerrainSystem *getSystem<TerrainSystem>() const
    {
        return getSystem_impl(&mAllocators.mTerrainSystemAllocator);
    }

    template <typename T> T *addSystem(size_t order)
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return nullptr;
    }

    template <> RenderSystem *addSystem<RenderSystem>(size_t order)
    {
        return addSystem_impl(&mAllocators.mRenderSystemAllocator, order);
    }

    template <> PhysicsSystem *addSystem<PhysicsSystem>(size_t order)
    {
        return addSystem_impl(&mAllocators.mPhysicsSystemAllocator, order);
    }
    template <> CleanUpSystem *addSystem<CleanUpSystem>(size_t order)
    {
        return addSystem_impl(&mAllocators.mCleanupSystemAllocator, order);
    }

    template <> DebugSystem *addSystem<DebugSystem>(size_t order)
    {
        return addSystem_impl(&mAllocators.mDebugSystemAllocator, order);
    }

    template <> GizmoSystem *addSystem<GizmoSystem>(size_t order)
    {
        return addSystem_impl(&mAllocators.mGizmoSystemAllocator, order);
    }

    template <> FreeLookCameraSystem *addSystem<FreeLookCameraSystem>(size_t order)
    {
        return addSystem_impl(&mAllocators.mFreeLookCameraSystemAllocator, order);
    }

    template <> TerrainSystem *addSystem<TerrainSystem>(size_t order)
    {
        return addSystem_impl(&mAllocators.mTerrainSystemAllocator, order);
    }

    template <typename T> T *getSystemByIndex(size_t index) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return nullptr;
    }

    template <> RenderSystem *getSystemByIndex<RenderSystem>(size_t index) const
    {
        return getSystemByIndex_impl(&mAllocators.mRenderSystemAllocator, index);
    }

    template <> PhysicsSystem *getSystemByIndex<PhysicsSystem>(size_t index) const
    {
        return getSystemByIndex_impl(&mAllocators.mPhysicsSystemAllocator, index);
    }
    template <> CleanUpSystem *getSystemByIndex<CleanUpSystem>(size_t index) const
    {
        return getSystemByIndex_impl(&mAllocators.mCleanupSystemAllocator, index);
    }

    template <> DebugSystem *getSystemByIndex<DebugSystem>(size_t index) const
    {
        return getSystemByIndex_impl(&mAllocators.mDebugSystemAllocator, index);
    }

    template <> GizmoSystem *getSystemByIndex<GizmoSystem>(size_t index) const
    {
        return getSystemByIndex_impl(&mAllocators.mGizmoSystemAllocator, index);
    }

    template <> FreeLookCameraSystem *getSystemByIndex<FreeLookCameraSystem>(size_t index) const
    {
        return getSystemByIndex_impl(&mAllocators.mFreeLookCameraSystemAllocator, index);
    }

    template <> TerrainSystem *getSystemByIndex<TerrainSystem>(size_t index) const
    {
        return getSystemByIndex_impl(&mAllocators.mTerrainSystemAllocator, index);
    }

    template <typename T> T *getSystemById(const Guid &systemId) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return nullptr;
    }

    template <> RenderSystem *getSystemById<RenderSystem>(const Guid &systemId) const
    {
        return getSystemById_impl(mIdState.mRenderSystemIdToGlobalIndex, &mAllocators.mRenderSystemAllocator, systemId);
    }

    template <> PhysicsSystem *getSystemById<PhysicsSystem>(const Guid &systemId) const
    {
        return getSystemById_impl(mIdState.mPhysicsSystemIdToGlobalIndex, &mAllocators.mPhysicsSystemAllocator,
                                  systemId);
    }
    template <> CleanUpSystem *getSystemById<CleanUpSystem>(const Guid &systemId) const
    {
        return getSystemById_impl(mIdState.mCleanupSystemIdToGlobalIndex, &mAllocators.mCleanupSystemAllocator,
                                  systemId);
    }

    template <> DebugSystem *getSystemById<DebugSystem>(const Guid &systemId) const
    {
        return getSystemById_impl(mIdState.mDebugSystemIdToGlobalIndex, &mAllocators.mDebugSystemAllocator, systemId);
    }

    template <> GizmoSystem *getSystemById<GizmoSystem>(const Guid &systemId) const
    {
        return getSystemById_impl(mIdState.mGizmoSystemIdToGlobalIndex, &mAllocators.mGizmoSystemAllocator, systemId);
    }

    template <> FreeLookCameraSystem *getSystemById<FreeLookCameraSystem>(const Guid &systemId) const
    {
        return getSystemById_impl(mIdState.mFreeLookCameraSystemIdToGlobalIndex, &mAllocators.mFreeLookCameraSystemAllocator,
                                  systemId);
    }

    template <> TerrainSystem *getSystemById<TerrainSystem>(const Guid &systemId) const
    {
        return getSystemById_impl(mIdState.mTerrainSystemIdToGlobalIndex, &mAllocators.mTerrainSystemAllocator,
                                  systemId);
    }

    template <typename T> T *getAssetByIndex(size_t index) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return nullptr;
    }

    template <> Mesh *getAssetByIndex<Mesh>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mMeshAllocator, index);
    }

    template <> Material *getAssetByIndex<Material>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mMaterialAllocator, index);
    }

    template <> Shader *getAssetByIndex<Shader>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mShaderAllocator, index);
    }

    template <> Texture2D *getAssetByIndex<Texture2D>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mTexture2DAllocator, index);
    }

    template <> Texture3D *getAssetByIndex<Texture3D>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mTexture3DAllocator, index);
    }

    template <> Cubemap *getAssetByIndex<Cubemap>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mCubemapAllocator, index);
    }

    template <> RenderTexture *getAssetByIndex<RenderTexture>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mRenderTextureAllocator, index);
    }

    template <> Font *getAssetByIndex<Font>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mFontAllocator, index);
    }

    template <> Sprite *getAssetByIndex<Sprite>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mSpriteAllocator, index);
    }

    template <typename T> T *getAssetById(const Guid &assetId) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return nullptr;
    }

    template <> Mesh *getAssetById<Mesh>(const Guid &assetId) const
    {
        return getAssetById_impl(mIdState.mMeshIdToGlobalIndex, &mAllocators.mMeshAllocator, assetId);
    }

    template <> Material *getAssetById<Material>(const Guid &assetId) const
    {
        return getAssetById_impl(mIdState.mMaterialIdToGlobalIndex, &mAllocators.mMaterialAllocator, assetId);
    }

    template <> Shader *getAssetById<Shader>(const Guid &assetId) const
    {
        return getAssetById_impl(mIdState.mShaderIdToGlobalIndex, &mAllocators.mShaderAllocator, assetId);
    }

    template <> Texture2D *getAssetById<Texture2D>(const Guid &assetId) const
    {
        return getAssetById_impl(mIdState.mTexture2DIdToGlobalIndex, &mAllocators.mTexture2DAllocator, assetId);
    }

    template <> Texture3D *getAssetById<Texture3D>(const Guid &assetId) const
    {
        return getAssetById_impl(mIdState.mTexture3DIdToGlobalIndex, &mAllocators.mTexture3DAllocator, assetId);
    }

    template <> Cubemap *getAssetById<Cubemap>(const Guid &assetId) const
    {
        return getAssetById_impl(mIdState.mCubemapIdToGlobalIndex, &mAllocators.mCubemapAllocator, assetId);
    }

    template <> RenderTexture *getAssetById<RenderTexture>(const Guid &assetId) const
    {
        return getAssetById_impl(mIdState.mRenderTextureIdToGlobalIndex, &mAllocators.mRenderTextureAllocator, assetId);
    }

    template <> Font *getAssetById<Font>(const Guid &assetId) const
    {
        return getAssetById_impl(mIdState.mFontIdToGlobalIndex, &mAllocators.mFontAllocator, assetId);
    }

    template <> Sprite *getAssetById<Sprite>(const Guid &assetId) const
    {
        return getAssetById_impl(mIdState.mSpriteIdToGlobalIndex, &mAllocators.mSpriteAllocator, assetId);
    }

    template <typename T> T *createAsset()
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return nullptr;
    }

    template <> Mesh *createAsset<Mesh>()
    {
        return createAsset_impl(&mAllocators.mMeshAllocator, Guid::newGuid());
    }

    template <> Material *createAsset<Material>()
    {
        return createAsset_impl(&mAllocators.mMaterialAllocator, Guid::newGuid());
    }

    template <> Shader *createAsset<Shader>()
    {
        return createAsset_impl(&mAllocators.mShaderAllocator, Guid::newGuid());
    }

    template <> Texture2D *createAsset<Texture2D>()
    {
        return createAsset_impl(&mAllocators.mTexture2DAllocator, Guid::newGuid());
    }

    template <> Texture3D *createAsset<Texture3D>()
    {
        return createAsset_impl(&mAllocators.mTexture3DAllocator, Guid::newGuid());
    }

    template <> Cubemap *createAsset<Cubemap>()
    {
        return createAsset_impl(&mAllocators.mCubemapAllocator, Guid::newGuid());
    }

    template <> RenderTexture *createAsset<RenderTexture>()
    {
        return createAsset_impl(&mAllocators.mRenderTextureAllocator, Guid::newGuid());
    }

    template <> Font *createAsset<Font>()
    {
        return createAsset_impl(&mAllocators.mFontAllocator, Guid::newGuid());
    }

    template <> Sprite *createAsset<Sprite>()
    {
        return createAsset_impl(&mAllocators.mSpriteAllocator, Guid::newGuid());
    }

    template <typename T> T *createAsset(const Guid& id)
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return nullptr;
    }

    template <> Mesh *createAsset<Mesh>(const Guid &id)
    {
        return createAsset_impl(&mAllocators.mMeshAllocator, id);
    }

    template <> Material *createAsset<Material>(const Guid &id)
    {
        return createAsset_impl(&mAllocators.mMaterialAllocator, id);
    }

    template <> Shader *createAsset<Shader>(const Guid &id)
    {
        return createAsset_impl(&mAllocators.mShaderAllocator, id);
    }

    template <> Texture2D *createAsset<Texture2D>(const Guid &id)
    {
        return createAsset_impl(&mAllocators.mTexture2DAllocator, id);
    }

    template <> Texture3D *createAsset<Texture3D>(const Guid &id)
    {
        return createAsset_impl(&mAllocators.mTexture3DAllocator, id);
    }

    template <> Cubemap *createAsset<Cubemap>(const Guid &id)
    {
        return createAsset_impl(&mAllocators.mCubemapAllocator, id);
    }

    template <> RenderTexture *createAsset<RenderTexture>(const Guid &id)
    {
        return createAsset_impl(&mAllocators.mRenderTextureAllocator, id);
    }

    template <> Font *createAsset<Font>(const Guid &id)
    {
        return createAsset_impl(&mAllocators.mFontAllocator, id);
    }

    template <> Sprite *createAsset<Sprite>(const Guid &id)
    {
        return createAsset_impl(&mAllocators.mSpriteAllocator, id);
    }

    template <typename T> T *createAsset(const YAML::Node &in)
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return nullptr;
    }

    template <> Mesh *createAsset<Mesh>(const YAML::Node &in)
    {
        return createAsset_impl(&mAllocators.mMeshAllocator, in);
    }

    template <> Material *createAsset<Material>(const YAML::Node &in)
    {
        return createAsset_impl(&mAllocators.mMaterialAllocator, in);
    }

    template <> Shader *createAsset<Shader>(const YAML::Node &in)
    {
        return createAsset_impl(&mAllocators.mShaderAllocator, in);
    }

    template <> Texture2D *createAsset<Texture2D>(const YAML::Node &in)
    {
        return createAsset_impl(&mAllocators.mTexture2DAllocator, in);
    }

    template <> Texture3D *createAsset<Texture3D>(const YAML::Node &in)
    {
        return createAsset_impl(&mAllocators.mTexture3DAllocator, in);
    }

    template <> Cubemap *createAsset<Cubemap>(const YAML::Node &in)
    {
        return createAsset_impl(&mAllocators.mCubemapAllocator, in);
    }

    template <> RenderTexture *createAsset<RenderTexture>(const YAML::Node &in)
    {
        return createAsset_impl(&mAllocators.mRenderTextureAllocator, in);
    }

    template <> Font *createAsset<Font>(const YAML::Node &in)
    {
        return createAsset_impl(&mAllocators.mFontAllocator, in);
    }

    template <> Sprite *createAsset<Sprite>(const YAML::Node &in)
    {
        return createAsset_impl(&mAllocators.mSpriteAllocator, in);
    }

  private:
    void generateSourcePaths(const std::string& filepath, YAML::Node &in);

    template <typename T> size_t getNumberOfSystems_impl(const PoolAllocator<T> *allocator) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return allocator != nullptr ? allocator->getCount() : 0;
    }

    template <typename T> size_t getNumberOfAssets_impl(const PoolAllocator<T> *allocator) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return allocator != nullptr ? allocator->getCount() : 0;
    }

    template <typename T> T *getSystem_impl(const PoolAllocator<T> *allocator) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return allocator != nullptr ? allocator->get(0) : nullptr;
    }

    template <typename T> T *addSystem_impl(PoolAllocator<T> *allocator, size_t order)
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        int systemGlobalIndex = (int)allocator->getCount();
        int systemType = SystemType<T>::type;

        T *system = allocator->construct(this, Guid::newGuid());

        if (system != nullptr)
        {
            addIdToGlobalIndexMap_impl<T>(system->getId(), systemGlobalIndex, systemType);

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

    template <typename T> T *getSystemByIndex_impl(const PoolAllocator<T> *allocator, size_t index) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return allocator != nullptr ? allocator->get(index) : nullptr;
    }

    template <typename T>
    T *getSystemById_impl(const std::unordered_map<Guid, int> &idToIndexMap, const PoolAllocator<T> *allocator,
                          const Guid &systemId) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        if (allocator == nullptr || SystemType<T>::type != getTypeOf(systemId))
        {
            return nullptr;
        }

        return getById_impl<T>(idToIndexMap, allocator, systemId);
    }

    template <typename T> T *getAssetByIndex_impl(const PoolAllocator<T> *allocator, size_t index) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return allocator != nullptr ? allocator->get(index) : nullptr;
    }

    template <typename T>
    T *getAssetById_impl(const std::unordered_map<Guid, int> &idToIndexMap, const PoolAllocator<T> *allocator,
                         const Guid &assetId) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        if (allocator == nullptr || AssetType<T>::type != getTypeOf(assetId))
        {
            return nullptr;
        }

        return getById_impl<T>(idToIndexMap, allocator, assetId);
    }

    template <typename T> T *createAsset_impl(PoolAllocator<T> *allocator, const Guid& id)
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        int index = (int)allocator->getCount();
        int type = AssetType<T>::type;

        T *asset = allocator->construct(this, id);

        if (asset != nullptr)
        {
            addIdToGlobalIndexMap_impl<T>(asset->getId(), index, type);
        }

        return asset;
    }

    template <typename T> T *createAsset_impl(PoolAllocator<T> *allocator, const YAML::Node &in)
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        int index = (int)allocator->getCount();
        int type = AssetType<T>::type;

        T *asset = allocator->construct(this, in);

        if (asset != nullptr)
        {
            addIdToGlobalIndexMap_impl<T>(asset->getId(), index, type);
        }

        return asset;
    }

    template <typename T>
    T *getById_impl(const std::unordered_map<Guid, int> &idToIndexMap, const PoolAllocator<T> *allocator,
                    const Guid &id) const
    {
        static_assert(std::is_base_of<Scene, T>() || std::is_base_of<Asset, T>() || std::is_base_of<System, T>(),
                      "'T' is not of type Asset or System");

        std::unordered_map<Guid, int>::const_iterator it = idToIndexMap.find(id);
        if (it != idToIndexMap.end())
        {
            return allocator->get(it->second);
        }
        else
        {
            return nullptr;
        }
    }

    template <typename T> void addIdToGlobalIndexMap_impl(const Guid &id, int index, int type)
    {
        static_assert(std::is_base_of<Scene, T>() || std::is_base_of<System, T>() || std::is_base_of<Asset, T>(),
                      "'T' is not of type Scene, System, or Asset");
    }

    template <> void addIdToGlobalIndexMap_impl<Scene>(const Guid &id, int index, int type)
    {
        mIdState.mSceneIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<Mesh>(const Guid &id, int index, int type)
    {
        mIdState.mMeshIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<Material>(const Guid &id, int index, int type)
    {
        mIdState.mMaterialIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<Shader>(const Guid &id, int index, int type)
    {
        mIdState.mShaderIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<Texture2D>(const Guid &id, int index, int type)
    {
        mIdState.mTexture2DIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<Texture3D>(const Guid &id, int index, int type)
    {
        mIdState.mTexture3DIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<Cubemap>(const Guid &id, int index, int type)
    {
        mIdState.mCubemapIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<RenderTexture>(const Guid &id, int index, int type)
    {
        mIdState.mRenderTextureIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<Font>(const Guid &id, int index, int type)
    {
        mIdState.mFontIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<Sprite>(const Guid &id, int index, int type)
    {
        mIdState.mSpriteIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<RenderSystem>(const Guid &id, int index, int type)
    {
        mIdState.mRenderSystemIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<PhysicsSystem>(const Guid &id, int index, int type)
    {
        mIdState.mPhysicsSystemIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<CleanUpSystem>(const Guid &id, int index, int type)
    {
        mIdState.mCleanupSystemIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<DebugSystem>(const Guid &id, int index, int type)
    {
        mIdState.mDebugSystemIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<GizmoSystem>(const Guid &id, int index, int type)
    {
        mIdState.mGizmoSystemIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<FreeLookCameraSystem>(const Guid &id, int index, int type)
    {
        mIdState.mFreeLookCameraSystemIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
    
    template <> void addIdToGlobalIndexMap_impl<TerrainSystem>(const Guid &id, int index, int type)
    {
        mIdState.mTerrainSystemIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
};
} // namespace PhysicsEngine

#endif