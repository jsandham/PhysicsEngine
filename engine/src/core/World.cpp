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

template <> size_t World::getNumberOfAssets<Texture3D>() const
{
    return mAllocators.mTexture3DAllocator.getCount();
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

template <> RenderSystem* World::getSystemById<RenderSystem>(const Guid& systemId) const
{
    return getSystemById_impl(mIdState.mRenderSystemIdToGlobalIndex, &mAllocators.mRenderSystemAllocator, systemId);
}

template <> PhysicsSystem* World::getSystemById<PhysicsSystem>(const Guid& systemId) const
{
    return getSystemById_impl(mIdState.mPhysicsSystemIdToGlobalIndex, &mAllocators.mPhysicsSystemAllocator,
        systemId);
}
template <> CleanUpSystem* World::getSystemById<CleanUpSystem>(const Guid& systemId) const
{
    return getSystemById_impl(mIdState.mCleanupSystemIdToGlobalIndex, &mAllocators.mCleanupSystemAllocator,
        systemId);
}

template <> DebugSystem* World::getSystemById<DebugSystem>(const Guid& systemId) const
{
    return getSystemById_impl(mIdState.mDebugSystemIdToGlobalIndex, &mAllocators.mDebugSystemAllocator, systemId);
}

template <> GizmoSystem* World::getSystemById<GizmoSystem>(const Guid& systemId) const
{
    return getSystemById_impl(mIdState.mGizmoSystemIdToGlobalIndex, &mAllocators.mGizmoSystemAllocator, systemId);
}

template <> FreeLookCameraSystem* World::getSystemById<FreeLookCameraSystem>(const Guid& systemId) const
{
    return getSystemById_impl(mIdState.mFreeLookCameraSystemIdToGlobalIndex, &mAllocators.mFreeLookCameraSystemAllocator,
        systemId);
}

template <> TerrainSystem* World::getSystemById<TerrainSystem>(const Guid& systemId) const
{
    return getSystemById_impl(mIdState.mTerrainSystemIdToGlobalIndex, &mAllocators.mTerrainSystemAllocator,
        systemId);
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

template <> Texture3D* World::getAssetByIndex<Texture3D>(size_t index) const
{
    return mAllocators.mTexture3DAllocator.get(index);
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

template <> Mesh* World::getAssetById<Mesh>(const Guid& assetId) const
{
    return getAssetById_impl(mIdState.mMeshIdToGlobalIndex, &mAllocators.mMeshAllocator, assetId);
}

template <> Material* World::getAssetById<Material>(const Guid& assetId) const
{
    return getAssetById_impl(mIdState.mMaterialIdToGlobalIndex, &mAllocators.mMaterialAllocator, assetId);
}

template <> Shader* World::getAssetById<Shader>(const Guid& assetId) const
{
    return getAssetById_impl(mIdState.mShaderIdToGlobalIndex, &mAllocators.mShaderAllocator, assetId);
}

template <> Texture2D* World::getAssetById<Texture2D>(const Guid& assetId) const
{
    return getAssetById_impl(mIdState.mTexture2DIdToGlobalIndex, &mAllocators.mTexture2DAllocator, assetId);
}

template <> Texture3D* World::getAssetById<Texture3D>(const Guid& assetId) const
{
    return getAssetById_impl(mIdState.mTexture3DIdToGlobalIndex, &mAllocators.mTexture3DAllocator, assetId);
}

template <> Cubemap* World::getAssetById<Cubemap>(const Guid& assetId) const
{
    return getAssetById_impl(mIdState.mCubemapIdToGlobalIndex, &mAllocators.mCubemapAllocator, assetId);
}

template <> RenderTexture* World::getAssetById<RenderTexture>(const Guid& assetId) const
{
    return getAssetById_impl(mIdState.mRenderTextureIdToGlobalIndex, &mAllocators.mRenderTextureAllocator, assetId);
}

template <> Font* World::getAssetById<Font>(const Guid& assetId) const
{
    return getAssetById_impl(mIdState.mFontIdToGlobalIndex, &mAllocators.mFontAllocator, assetId);
}

template <> Sprite* World::getAssetById<Sprite>(const Guid& assetId) const
{
    return getAssetById_impl(mIdState.mSpriteIdToGlobalIndex, &mAllocators.mSpriteAllocator, assetId);
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

template <> Texture3D* World::createAsset<Texture3D>()
{
    return createAsset_impl(&mAllocators.mTexture3DAllocator, Guid::newGuid());
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

template <> Mesh* World::createAsset<Mesh>(const Guid& id)
{
    return createAsset_impl(&mAllocators.mMeshAllocator, id);
}

template <> Material* World::createAsset<Material>(const Guid& id)
{
    return createAsset_impl(&mAllocators.mMaterialAllocator, id);
}

template <> Shader* World::createAsset<Shader>(const Guid& id)
{
    return createAsset_impl(&mAllocators.mShaderAllocator, id);
}

template <> Texture2D* World::createAsset<Texture2D>(const Guid& id)
{
    return createAsset_impl(&mAllocators.mTexture2DAllocator, id);
}

template <> Texture3D* World::createAsset<Texture3D>(const Guid& id)
{
    return createAsset_impl(&mAllocators.mTexture3DAllocator, id);
}

template <> Cubemap* World::createAsset<Cubemap>(const Guid& id)
{
    return createAsset_impl(&mAllocators.mCubemapAllocator, id);
}

template <> RenderTexture* World::createAsset<RenderTexture>(const Guid& id)
{
    return createAsset_impl(&mAllocators.mRenderTextureAllocator, id);
}

template <> Font* World::createAsset<Font>(const Guid& id)
{
    return createAsset_impl(&mAllocators.mFontAllocator, id);
}

template <> Sprite* World::createAsset<Sprite>(const Guid& id)
{
    return createAsset_impl(&mAllocators.mSpriteAllocator, id);
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

template <> Texture3D* World::createAsset<Texture3D>(const YAML::Node& in)
{
    return createAsset_impl(&mAllocators.mTexture3DAllocator, in);
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

template <> void World::addIdToGlobalIndexMap_impl<Scene>(const Guid& id, int index, int type)
{
    mIdState.mSceneIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Mesh>(const Guid& id, int index, int type)
{
    mIdState.mMeshIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Material>(const Guid& id, int index, int type)
{
    mIdState.mMaterialIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Shader>(const Guid& id, int index, int type)
{
    mIdState.mShaderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Texture2D>(const Guid& id, int index, int type)
{
    mIdState.mTexture2DIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Texture3D>(const Guid& id, int index, int type)
{
    mIdState.mTexture3DIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Cubemap>(const Guid& id, int index, int type)
{
    mIdState.mCubemapIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<RenderTexture>(const Guid& id, int index, int type)
{
    mIdState.mRenderTextureIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Font>(const Guid& id, int index, int type)
{
    mIdState.mFontIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Sprite>(const Guid& id, int index, int type)
{
    mIdState.mSpriteIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<RenderSystem>(const Guid& id, int index, int type)
{
    mIdState.mRenderSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<PhysicsSystem>(const Guid& id, int index, int type)
{
    mIdState.mPhysicsSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<CleanUpSystem>(const Guid& id, int index, int type)
{
    mIdState.mCleanupSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<DebugSystem>(const Guid& id, int index, int type)
{
    mIdState.mDebugSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<GizmoSystem>(const Guid& id, int index, int type)
{
    mIdState.mGizmoSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<FreeLookCameraSystem>(const Guid& id, int index, int type)
{
    mIdState.mFreeLookCameraSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<TerrainSystem>(const Guid& id, int index, int type)
{
    mIdState.mTerrainSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <typename T> T* World::addSystem_impl(PoolAllocator<T>* allocator, size_t order)
{
    static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

    int systemGlobalIndex = (int)allocator->getCount();
    int systemType = SystemType<T>::type;

    T* system = allocator->construct(this, Guid::newGuid());

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

template <typename T>
T* World::getSystemById_impl(const std::unordered_map<Guid, int>& idToIndexMap, const PoolAllocator<T>* allocator,
    const Guid& systemId) const
{
    static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

    if (allocator == nullptr || SystemType<T>::type != getTypeOf(systemId))
    {
        return nullptr;
    }

    return getById_impl<T>(idToIndexMap, allocator, systemId);
}

template <typename T>
T* World::getAssetById_impl(const std::unordered_map<Guid, int>& idToIndexMap, const PoolAllocator<T>* allocator,
    const Guid& assetId) const
{
    static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

    if (allocator == nullptr || AssetType<T>::type != getTypeOf(assetId))
    {
        return nullptr;
    }

    return getById_impl<T>(idToIndexMap, allocator, assetId);
}

template <typename T> T* World::createAsset_impl(PoolAllocator<T>* allocator, const Guid& assetId)
{
    static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

    int index = (int)allocator->getCount();
    int type = AssetType<T>::type;

    T* asset = allocator->construct(this, assetId);

    if (asset != nullptr)
    {
        addIdToGlobalIndexMap_impl<T>(asset->getId(), index, type);
    }

    return asset;
}

template <typename T> T* World::createAsset_impl(PoolAllocator<T>* allocator, const YAML::Node& in)
{
    static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

    int index = (int)allocator->getCount();
    int type = AssetType<T>::type;

    T* asset = allocator->construct(this, in);

    if (asset != nullptr)
    {
        addIdToGlobalIndexMap_impl<T>(asset->getId(), index, type);
    }

    return asset;
}

template <typename T>
T* World::getById_impl(const std::unordered_map<Guid, int>& idToIndexMap, const PoolAllocator<T>* allocator,
    const Guid& id) const
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
        Guid id = YAML::getValue<Guid>(in.begin()->second, "id");

        if (PhysicsEngine::isAsset(type) && id.isValid())
        {
            generateSourcePaths(filePath, in.begin()->second);

            Asset *asset = getAssetById(id, type);
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
                mIdState.mAssetIdToFilepath[asset->getId()] = filePath;
                mIdState.mAssetFilepathToId[filePath] = asset->getId();
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
    Guid id = YAML::getValue<Guid>(in, "id");

    if (PhysicsEngine::isScene(type) && id.isValid())
    {
        Scene *scene = getSceneById(id);
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
            mIdState.mSceneIdToFilepath[scene->getId()] = filePath;
            mIdState.mSceneFilepathToId[filePath] = scene->getId();

            // Copy 'do not destroy' entities from old scene to new scene
            copyDoNotDestroyEntities(mActiveScene, scene);

            mActiveScene = scene;
        }

        return scene;
    }

    return nullptr;
}

bool World::writeAssetToYAML(const std::string &filePath, const Guid &assetId) const
{
    int type = getTypeOf(assetId);

    Asset *asset = getAssetById(assetId, type);
    if (asset == nullptr)
    {
        return false;    
    }

    return asset->writeToYAML(filePath);
}

bool World::writeSceneToYAML(const std::string &filePath, const Guid &sceneId) const
{
    Scene *scene = getSceneById(sceneId);
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

            std::cout << "do not destroy entity: " << entity->getId().toString() << std::endl;

            Entity *newEntity = to->getEntityById(entity->getId());
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
                Component *component = from->getComponentById(components[j].first, components[j].second);

                YAML::Node componentNode;
                component->serialize(componentNode);

                Component *newComponent = to->getComponentById(components[j].first, components[j].second);
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

std::vector<ShaderUniform> World::getCachedMaterialUniforms(const Guid &materialId, const Guid &shaderId)
{
    return mMaterialUniformCache[materialId][shaderId];
}

void World::cacheMaterialUniforms(const Guid &materialId, const Guid &shaderId, const std::vector<ShaderUniform> &uniforms)
{
    assert(materialId != Guid::INVALID);
    assert(shaderId != Guid::INVALID);

    mMaterialUniformCache[materialId][shaderId] = uniforms;
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
        return getAssetById<Mesh>(mPrimitives.mPlaneMeshId);
    case PrimitiveType::Disc:
        return getAssetById<Mesh>(mPrimitives.mDiscMeshId);
    case PrimitiveType::Cube:
        return getAssetById<Mesh>(mPrimitives.mCubeMeshId);
    case PrimitiveType::Sphere:
        return getAssetById<Mesh>(mPrimitives.mSphereMeshId);
    case PrimitiveType::Cylinder:
        return getAssetById<Mesh>(mPrimitives.mCylinderMeshId);
    case PrimitiveType::Cone:
        return getAssetById<Mesh>(mPrimitives.mConeMeshId);
    default:
        return nullptr;
    }
}


Material *World::getPrimtiveMaterial() const
{
    return getAssetById<Material>(mPrimitives.mStandardMaterialId);
}

Asset *World::getAssetById(const Guid &assetId, int type) const
{
    switch (type)
    {
    case AssetType<Mesh>::type: {return getAssetById<Mesh>(assetId);}
    case AssetType<Material>::type: {return getAssetById<Material>(assetId);}
    case AssetType<Shader>::type: {return getAssetById<Shader>(assetId);}
    case AssetType<Texture2D>::type: {return getAssetById<Texture2D>(assetId);}
    case AssetType<Texture3D>::type: {return getAssetById<Texture3D>(assetId);}
    case AssetType<Cubemap>::type: {return getAssetById<Cubemap>(assetId);}
    case AssetType<RenderTexture>::type: {return getAssetById<RenderTexture>(assetId);}
    case AssetType<Sprite>::type: {return getAssetById<Sprite>(assetId);}
    case AssetType<Font>::type: {return getAssetById<Font>(assetId);}
    }

    return nullptr;
}

Scene *World::getSceneById(const Guid &sceneId) const
{
    return getById_impl<Scene>(mIdState.mSceneIdToGlobalIndex, &mAllocators.mSceneAllocator, sceneId);
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

int World::getIndexOf(const Guid &id) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mIdToGlobalIndex.find(id);
    if (it != mIdState.mIdToGlobalIndex.end())
    {
        return it->second;
    }

    return -1;
}

int World::getTypeOf(const Guid &id) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mIdToType.find(id);
    if (it != mIdState.mIdToType.end())
    {
        return it->second;
    }

    return -1;
}

Scene *World::createScene()
{
    int globalIndex = (int)mAllocators.mSceneAllocator.getCount();
    int type = SceneType<Scene>::type;

    Scene *scene = mAllocators.mSceneAllocator.construct(this, Guid::newGuid());

    if (scene != nullptr)
    {
        addIdToGlobalIndexMap_impl<Scene>(scene->getId(), globalIndex, type);
    }

    return scene;
}

Scene *World::createScene(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mSceneAllocator.getCount();
    int type = SceneType<Scene>::type;

    Scene *scene = mAllocators.mSceneAllocator.construct(this, in);

    if (scene != nullptr)
    {
        addIdToGlobalIndexMap_impl<Scene>(scene->getId(), globalIndex, type);
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
    case AssetType<Texture3D>::type: {
        return createAsset<Texture3D>(in);
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

void World::latentDestroyAsset(const Guid &assetId, int assetType)
{
    mIdState.mAssetIdsMarkedLatentDestroy.push_back(std::make_pair(assetId, assetType));
}

void World::immediateDestroyAsset(const Guid &assetId, int assetType)
{
    int index = getIndexOf(assetId);
    Asset *swap = nullptr;
    
    if (assetType == AssetType<Material>::type)
    {
        swap = mAllocators.mMaterialAllocator.destruct(index);
    
        mIdState.mMaterialIdToGlobalIndex.erase(assetId);
        mIdState.mIdToGlobalIndex.erase(assetId);
        mIdState.mIdToType.erase(assetId);
    
        if (swap != nullptr)
        {
            mIdState.mMaterialIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = assetType;
        }
    }
    else if (assetType == AssetType<Mesh>::type)
    {
        swap = mAllocators.mMeshAllocator.destruct(index);
    
        mIdState.mMeshIdToGlobalIndex.erase(assetId);
        mIdState.mIdToGlobalIndex.erase(assetId);
        mIdState.mIdToType.erase(assetId);
    
        if (swap != nullptr)
        {
            mIdState.mMeshIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = assetType;
        }
    }
    else if (assetType == AssetType<Shader>::type)
    {
        swap = mAllocators.mShaderAllocator.destruct(index);
    
        mIdState.mShaderIdToGlobalIndex.erase(assetId);
        mIdState.mIdToGlobalIndex.erase(assetId);
        mIdState.mIdToType.erase(assetId);
    
        if (swap != nullptr)
        {
            mIdState.mShaderIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = assetType;
        }
    }
    else if (assetType == AssetType<Texture2D>::type)
    {
        swap = mAllocators.mTexture2DAllocator.destruct(index);
    
        mIdState.mTexture2DIdToGlobalIndex.erase(assetId);
        mIdState.mIdToGlobalIndex.erase(assetId);
        mIdState.mIdToType.erase(assetId);
    
        if (swap != nullptr)
        {
            mIdState.mTexture2DIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = assetType;
        }
    }
    else if (assetType == AssetType<Texture3D>::type)
    {
        swap = mAllocators.mTexture3DAllocator.destruct(index);
    
        mIdState.mTexture3DIdToGlobalIndex.erase(assetId);
        mIdState.mIdToGlobalIndex.erase(assetId);
        mIdState.mIdToType.erase(assetId);
    
        if (swap != nullptr)
        {
            mIdState.mTexture3DIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = assetType;
        }
    }
    else if (assetType == AssetType<Cubemap>::type)
    {
        swap = mAllocators.mCubemapAllocator.destruct(index);
    
        mIdState.mCubemapIdToGlobalIndex.erase(assetId);
        mIdState.mIdToGlobalIndex.erase(assetId);
        mIdState.mIdToType.erase(assetId);
    
        if (swap != nullptr)
        {
            mIdState.mCubemapIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = assetType;
        }
    }
    else if (assetType == AssetType<RenderTexture>::type)
    {
        swap = mAllocators.mRenderTextureAllocator.destruct(index);
    
        mIdState.mRenderTextureIdToGlobalIndex.erase(assetId);
        mIdState.mIdToGlobalIndex.erase(assetId);
        mIdState.mIdToType.erase(assetId);
    
        if (swap != nullptr)
        {
            mIdState.mRenderTextureIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = assetType;
        }
    }
    else if (assetType == AssetType<Font>::type)
    {
        swap = mAllocators.mFontAllocator.destruct(index);
    
        mIdState.mFontIdToGlobalIndex.erase(assetId);
        mIdState.mIdToGlobalIndex.erase(assetId);
        mIdState.mIdToType.erase(assetId);
    
        if (swap != nullptr)
        {
            mIdState.mFontIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = assetType;
        }
    }
    else if (assetType == AssetType<Sprite>::type)
    {
        swap = mAllocators.mSpriteAllocator.destruct(index);
    
        mIdState.mSpriteIdToGlobalIndex.erase(assetId);
        mIdState.mIdToGlobalIndex.erase(assetId);
        mIdState.mIdToType.erase(assetId);
    
        if (swap != nullptr)
        {
            mIdState.mSpriteIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = assetType;
        }
    }
    else
    {
        std::string message = "Error: Invalid asset instance type (" + std::to_string(assetType) +
                                ") when trying to destroy internal asset\n";
        Log::error(message.c_str());
    }
}

std::string World::getAssetFilepath(const Guid &assetId) const
{
    std::unordered_map<Guid, std::string>::const_iterator it = mIdState.mAssetIdToFilepath.find(assetId);
    if (it != mIdState.mAssetIdToFilepath.end())
    {
        return it->second;
    }

    return std::string();
}

std::string World::getSceneFilepath(const Guid &sceneId) const
{
    std::unordered_map<Guid, std::string>::const_iterator it = mIdState.mSceneIdToFilepath.find(sceneId);
    if (it != mIdState.mSceneIdToFilepath.end())
    {
        return it->second;
    }

    return std::string();
}

Guid World::getAssetId(const std::string& filepath) const
{
    std::unordered_map<std::string, Guid>::const_iterator it = mIdState.mAssetFilepathToId.find(filepath);
    if (it != mIdState.mAssetFilepathToId.end())
    {
        return it->second;
    }

    return Guid::INVALID;
}

Guid World::getSceneId(const std::string& filepath) const
{
    std::unordered_map<std::string, Guid>::const_iterator it = mIdState.mSceneFilepathToId.find(filepath);
    if (it != mIdState.mSceneFilepathToId.end())
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