#include <assert.h>
#include <fstream>
#include <stack>

#include "../../include/core/AssetTypes.h"
#include "../../include/core/Log.h"
#include "../../include/core/World.h"

#include "../../include/systems/SystemTypes.h"

#include "../../include/components/ComponentTypes.h"

#include "../../include/graphics/RenderContext.h"

using namespace PhysicsEngine;

template <typename T> static void copyComponentFromSceneToScene(const Scene *from, Scene *to, const Guid &guid)
{
    static_assert(IsComponent<T>::value);

    T *component = from->getComponentByGuid<T>(guid);

    YAML::Node componentNode;
    component->serialize(componentNode);

    T *newComponent = to->getComponentByGuid<T>(guid);
    if (newComponent != nullptr)
    {
        newComponent->deserialize(componentNode);
    }
    else
    {
        to->addComponent<T>(componentNode);
    }
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

template <> RenderSystem *World::getSystem<RenderSystem>() const
{
    return mRenderSystem;
}

template <> PhysicsSystem *World::getSystem<PhysicsSystem>() const
{
    return mPhysicsSystem;
}
template <> CleanUpSystem *World::getSystem<CleanUpSystem>() const
{
    return mCleanUpSystem;
}

template <> DebugSystem *World::getSystem<DebugSystem>() const
{
    return mDebugSystem;
}

template <> GizmoSystem *World::getSystem<GizmoSystem>() const
{
    return mGizmoSystem;
}

template <> FreeLookCameraSystem *World::getSystem<FreeLookCameraSystem>() const
{
    return mFreeLookCameraSystem;
}

template <> TerrainSystem *World::getSystem<TerrainSystem>() const
{
    return mTerrainSystem;
}

template <> AssetLoadingSystem *World::getSystem<AssetLoadingSystem>() const
{
    return mAssetLoadingSystem;
}

template <> Mesh *World::getAssetByIndex<Mesh>(size_t index) const
{
    return mAllocators.mMeshAllocator.get(index);
}

template <> Material *World::getAssetByIndex<Material>(size_t index) const
{
    return mAllocators.mMaterialAllocator.get(index);
}

template <> Shader *World::getAssetByIndex<Shader>(size_t index) const
{
    return mAllocators.mShaderAllocator.get(index);
}

template <> Texture2D *World::getAssetByIndex<Texture2D>(size_t index) const
{
    return mAllocators.mTexture2DAllocator.get(index);
}

template <> Cubemap *World::getAssetByIndex<Cubemap>(size_t index) const
{
    return mAllocators.mCubemapAllocator.get(index);
}

template <> RenderTexture *World::getAssetByIndex<RenderTexture>(size_t index) const
{
    return mAllocators.mRenderTextureAllocator.get(index);
}

template <> Mesh *World::getAssetById<Mesh>(const Id &assetId) const
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

template <> Mesh *World::createAsset<Mesh>()
{
    return createAsset_impl(&mAllocators.mMeshAllocator, Guid::newGuid());
}

template <> Material *World::createAsset<Material>()
{
    return createAsset_impl(&mAllocators.mMaterialAllocator, Guid::newGuid());
}

template <> Shader *World::createAsset<Shader>()
{
    return createAsset_impl(&mAllocators.mShaderAllocator, Guid::newGuid());
}

template <> Texture2D *World::createAsset<Texture2D>()
{
    return createAsset_impl(&mAllocators.mTexture2DAllocator, Guid::newGuid());
}

template <> Cubemap *World::createAsset<Cubemap>()
{
    return createAsset_impl(&mAllocators.mCubemapAllocator, Guid::newGuid());
}

template <> RenderTexture *World::createAsset<RenderTexture>()
{
    return createAsset_impl(&mAllocators.mRenderTextureAllocator, Guid::newGuid());
}

template <> Mesh *World::createAsset<Mesh>(const Guid &assetGuid)
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

template <> Mesh *World::createAsset<Mesh>(const YAML::Node &in)
{
    return createAsset_impl(&mAllocators.mMeshAllocator, in);
}

template <> Material *World::createAsset<Material>(const YAML::Node &in)
{
    return createAsset_impl(&mAllocators.mMaterialAllocator, in);
}

template <> Shader *World::createAsset<Shader>(const YAML::Node &in)
{
    return createAsset_impl(&mAllocators.mShaderAllocator, in);
}

template <> Texture2D *World::createAsset<Texture2D>(const YAML::Node &in)
{
    return createAsset_impl(&mAllocators.mTexture2DAllocator, in);
}

template <> Cubemap *World::createAsset<Cubemap>(const YAML::Node &in)
{
    return createAsset_impl(&mAllocators.mCubemapAllocator, in);
}

template <> RenderTexture *World::createAsset<RenderTexture>(const YAML::Node &in)
{
    return createAsset_impl(&mAllocators.mRenderTextureAllocator, in);
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

template <typename T>
T *World::getAssetById_impl(const std::unordered_map<Id, int> &idToIndexMap, const PoolAllocator<T> *allocator,
                            const Id &assetId) const
{
    static_assert(IsAsset<T>::value, "'T' is not of type Asset");

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
    static_assert(IsAsset<T>::value, "'T' is not of type Asset");

    if (allocator == nullptr || AssetType<T>::type != getTypeOf(assetGuid))
    {
        return nullptr;
    }

    return getByGuid_impl<T>(guidToIndexMap, allocator, assetGuid);
}

template <typename T> T *World::createAsset_impl(PoolAllocator<T> *allocator, const Guid &assetGuid)
{
    static_assert(IsAsset<T>::value, "'T' is not of type Asset");

    int index = (int)allocator->getCount();
    int type = AssetType<T>::type;

    T *asset = allocator->construct(this, assetGuid, Id::newId());

    if (asset != nullptr)
    {
        addToIdState_impl<T>(asset->getGuid(), asset->getId(), index, type);
    }

    return asset;
}

template <typename T> T *World::createAsset_impl(PoolAllocator<T> *allocator, const YAML::Node &in)
{
    static_assert(IsAsset<T>::value, "'T' is not of type Asset");

    int index = (int)allocator->getCount();
    int type = AssetType<T>::type;

    T *asset = allocator->construct(this, in, Id::newId());

    if (asset != nullptr)
    {
        addToIdState_impl<T>(asset->getGuid(), asset->getId(), index, type);
    }

    return asset;
}

template <typename T>
T *World::getById_impl(const std::unordered_map<Id, int> &idToIndexMap, const PoolAllocator<T> *allocator,
                       const Id &id) const
{
    static_assert(std::is_same<Scene, T>::value || IsAsset<T>::value || IsSystem<T>::value,
                  "'T' is not of type Scene, Asset or System");

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
    static_assert(std::is_same<Scene, T>::value || IsAsset<T>::value || IsSystem<T>::value,
                  "'T' is not of type Scene, Asset or System");

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
    mAssetLoadingSystem = new AssetLoadingSystem(this, Guid::newGuid(), Id::newId());
    mCleanUpSystem = new CleanUpSystem(this, Guid::newGuid(), Id::newId());
    mDebugSystem = new DebugSystem(this, Guid::newGuid(), Id::newId());
    mFreeLookCameraSystem = new FreeLookCameraSystem(this, Guid::newGuid(), Id::newId());
    mGizmoSystem = new GizmoSystem(this, Guid::newGuid(), Id::newId());
    mPhysicsSystem = new PhysicsSystem(this, Guid::newGuid(), Id::newId());
    mRenderSystem = new RenderSystem(this, Guid::newGuid(), Id::newId());
    mTerrainSystem = new TerrainSystem(this, Guid::newGuid(), Id::newId());

    mActiveScene = createScene();
    mPrimitives.createPrimitiveMeshes(this, 10, 10);
}

World::~World()
{
    delete mAssetLoadingSystem;
    delete mCleanUpSystem;
    delete mDebugSystem;
    delete mFreeLookCameraSystem;
    delete mGizmoSystem;
    delete mPhysicsSystem;
    delete mRenderSystem;
    delete mTerrainSystem;
}

void World::loadAllAssetsInPath(const std::filesystem::path &filePath)
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

                    std::filesystem::path relativeDataPath =
                        entry.path().lexically_relative(std::filesystem::current_path());

                    std::cout << "relative data path: " << relativeDataPath.string() << std::endl;
                    if (extension == MESH_EXT)
                    {
                        loadMeshFromYAML(relativeDataPath.string());
                    }
                    else if (extension == SHADER_EXT)
                    {
                        loadShaderFromYAML(relativeDataPath.string());
                    }
                    else if (extension == MATERIAL_EXT)
                    {
                        loadMaterialFromYAML(relativeDataPath.string());
                    }
                    else if (extension == TEXTURE2D_EXT)
                    {
                        loadTexture2DFromYAML(relativeDataPath.string());
                    }
                }
            }
        }
    }
}

bool World::loadAssetYAML(const std::string &filePath, YAML::Node &in, Guid &guid, int &type)
{
    try
    {
        in = YAML::LoadFile(filePath);
    }
    catch (YAML::Exception e /*YAML::BadFile e*/)
    {
        Log::error("YAML exception hit when trying to load file");
        return false;
    }

    if (!in.IsMap() || in.begin() == in.end())
    {
        return false;
    }

    if (in.begin()->first.IsScalar() && in.begin()->second.IsMap())
    {
        type = YAML::getValue<int>(in.begin()->second, "type");
        guid = YAML::getValue<Guid>(in.begin()->second, "id");

        if (PhysicsEngine::isAsset(type) && guid.isValid())
        {
            generateSourcePaths(filePath, in.begin()->second);
            return true;
        }
    }

    return false;
}

Cubemap *World::loadCubemapFromYAML(const std::string &filePath)
{
    YAML::Node in;

    Guid guid = Guid::INVALID;
    int type = -1;
    if (loadAssetYAML(filePath, in, guid, type))
    {
        Cubemap *cubemap = getAssetByGuid<Cubemap>(guid);
        if (cubemap != nullptr)
        {
            cubemap->deserialize(in.begin()->second);
        }
        else
        {
            cubemap = createAsset<Cubemap>(in.begin()->second);
        }

        return cubemap;
    }

    return nullptr;
}

Material *World::loadMaterialFromYAML(const std::string &filePath)
{
    YAML::Node in;

    Guid guid = Guid::INVALID;
    int type = -1;
    if (loadAssetYAML(filePath, in, guid, type))
    {
        Material *material = getAssetByGuid<Material>(guid);
        if (material != nullptr)
        {
            material->deserialize(in.begin()->second);
        }
        else
        {
            material = createAsset<Material>(in.begin()->second);
        }

        return material;
    }

    return nullptr;
}

Mesh *World::loadMeshFromYAML(const std::string &filePath)
{
    YAML::Node in;

    Guid guid = Guid::INVALID;
    int type = -1;
    if (loadAssetYAML(filePath, in, guid, type))
    {
        Mesh *mesh = getAssetByGuid<Mesh>(guid);
        if (mesh != nullptr)
        {
            mesh->deserialize(in.begin()->second);
        }
        else
        {
            mesh = createAsset<Mesh>(in.begin()->second);
        }

        return mesh;
    }

    return nullptr;
}

RenderTexture *World::loadRenderTextureFromYAML(const std::string &filePath)
{
    YAML::Node in;

    Guid guid = Guid::INVALID;
    int type = -1;
    if (loadAssetYAML(filePath, in, guid, type))
    {
        RenderTexture *texture = getAssetByGuid<RenderTexture>(guid);
        if (texture != nullptr)
        {
            texture->deserialize(in.begin()->second);
        }
        else
        {
            texture = createAsset<RenderTexture>(in.begin()->second);
        }

        return texture;
    }

    return nullptr;
}

Shader *World::loadShaderFromYAML(const std::string &filePath)
{
    YAML::Node in;

    Guid guid = Guid::INVALID;
    int type = -1;
    if (loadAssetYAML(filePath, in, guid, type))
    {
        Shader *shader = getAssetByGuid<Shader>(guid);
        if (shader != nullptr)
        {
            shader->deserialize(in.begin()->second);
        }
        else
        {
            shader = createAsset<Shader>(in.begin()->second);
        }

        return shader;
    }

    return nullptr;
}

Texture2D *World::loadTexture2DFromYAML(const std::string &filePath)
{
    YAML::Node in;

    Guid guid = Guid::INVALID;
    int type = -1;
    if (loadAssetYAML(filePath, in, guid, type))
    {
        Texture2D *texture = getAssetByGuid<Texture2D>(guid);
        if (texture != nullptr)
        {
            texture->deserialize(in.begin()->second);
        }
        else
        {
            texture = createAsset<Texture2D>(in.begin()->second);
        }

        return texture;
    }

    return nullptr;
}

Material *loadMaterialFromYAML(const std::string &filePath);
Mesh *loadMeshFromYAML(const std::string &filePath);
RenderTexture *loadRenderTextureFromYAML(const std::string &filePath);
Texture2D *loadTexture2DFromYAML(const std::string &filePath);

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

    switch (type)
    {
    case AssetType<Cubemap>::type: {
        Cubemap *cubemap = getCubemapByGuid(assetGuid);
        if (cubemap == nullptr)
        {
            return false;
        }

        return cubemap->writeToYAML(filePath);
    }
    case AssetType<Material>::type: {
        Material *material = getMaterialByGuid(assetGuid);
        if (material == nullptr)
        {
            return false;
        }

        return material->writeToYAML(filePath);
    }
    case AssetType<Mesh>::type: {
        Mesh *mesh = getMeshByGuid(assetGuid);
        if (mesh == nullptr)
        {
            return false;
        }

        return mesh->writeToYAML(filePath);
    }
    case AssetType<RenderTexture>::type: {
        RenderTexture *texture = getRenderTextureByGuid(assetGuid);
        if (texture == nullptr)
        {
            return false;
        }

        return texture->writeToYAML(filePath);
    }
    case AssetType<Shader>::type: {
        Shader *shader = getShaderByGuid(assetGuid);
        if (shader == nullptr)
        {
            return false;
        }

        return shader->writeToYAML(filePath);
    }
    case AssetType<Texture2D>::type: {
        Texture2D *texture = getTexture2DByGuid(assetGuid);
        if (texture == nullptr)
        {
            return false;
        }

        return texture->writeToYAML(filePath);
    }
    }

    return false;
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
                switch (components[j].second)
                {
                case ComponentType<Transform>::type: {
                    copyComponentFromSceneToScene<Transform>(from, to, components[j].first);
                    break;
                }
                case ComponentType<Rigidbody>::type: {
                    copyComponentFromSceneToScene<Rigidbody>(from, to, components[j].first);
                    break;
                }
                case ComponentType<Camera>::type: {
                    copyComponentFromSceneToScene<Camera>(from, to, components[j].first);
                    break;
                }
                case ComponentType<MeshRenderer>::type: {
                    copyComponentFromSceneToScene<MeshRenderer>(from, to, components[j].first);
                    break;
                }
                case ComponentType<LineRenderer>::type: {
                    copyComponentFromSceneToScene<LineRenderer>(from, to, components[j].first);
                    break;
                }
                case ComponentType<Light>::type: {
                    copyComponentFromSceneToScene<Light>(from, to, components[j].first);
                    break;
                }
                case ComponentType<BoxCollider>::type: {
                    copyComponentFromSceneToScene<BoxCollider>(from, to, components[j].first);
                    break;
                }
                case ComponentType<SphereCollider>::type: {
                    copyComponentFromSceneToScene<SphereCollider>(from, to, components[j].first);
                    break;
                }
                case ComponentType<MeshCollider>::type: {
                    copyComponentFromSceneToScene<MeshCollider>(from, to, components[j].first);
                    break;
                }
                case ComponentType<CapsuleCollider>::type: {
                    copyComponentFromSceneToScene<CapsuleCollider>(from, to, components[j].first);
                    break;
                }
                case ComponentType<Terrain>::type: {
                    copyComponentFromSceneToScene<Terrain>(from, to, components[j].first);
                    break;
                }
                default:
                    assert(!"Unreachable code");
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
        case AssetType<Shader>::type: {
            switch (RenderContext::getRenderAPI())
            {
            case RenderAPI::OpenGL: {
                std::filesystem::path source = YAML::getValue<std::string>(in, "source");
                in["sourceFilepath"] = (path / source).string();
                break;
            }
            case RenderAPI::DirectX: {
                std::filesystem::path source = YAML::getValue<std::string>(in, "hlsl_source");
                in["sourceFilepath"] = (path / source).string();
                break;
            }
            }
            break;
        }
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

void World::cacheMaterialUniforms(const Guid &materialGuid, const Guid &shaderGuid,
                                  const std::vector<ShaderUniform> &uniforms)
{
    assert(materialGuid != Guid::INVALID);
    assert(shaderGuid != Guid::INVALID);

    mMaterialUniformCache[materialGuid][shaderGuid] = uniforms;
}

size_t World::getNumberOfScenes() const
{
    return mAllocators.mSceneAllocator.getCount();
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

Cubemap *World::getCubemapById(const Id &assetId) const
{
    return getAssetById<Cubemap>(assetId);
}

Material *World::getMaterialById(const Id &assetId) const
{
    return getAssetById<Material>(assetId);
}

Mesh *World::getMeshById(const Id &assetId) const
{
    return getAssetById<Mesh>(assetId);
}

RenderTexture *World::getRenderTexutreById(const Id &assetId) const
{
    return getAssetById<RenderTexture>(assetId);
}

Shader *World::getShaderById(const Id &assetId) const
{
    return getAssetById<Shader>(assetId);
}

Texture2D *World::getTexture2DById(const Id &assetId) const
{
    return getAssetById<Texture2D>(assetId);
}

Cubemap *World::getCubemapByGuid(const Guid &assetGuid) const
{
    return getAssetByGuid<Cubemap>(assetGuid);
}

Material *World::getMaterialByGuid(const Guid &assetGuid) const
{
    return getAssetByGuid<Material>(assetGuid);
}

Mesh *World::getMeshByGuid(const Guid &assetGuid) const
{
    return getAssetByGuid<Mesh>(assetGuid);
}

RenderTexture *World::getRenderTextureByGuid(const Guid &assetGuid) const
{
    return getAssetByGuid<RenderTexture>(assetGuid);
}

Shader *World::getShaderByGuid(const Guid &assetGuid) const
{
    return getAssetByGuid<Shader>(assetGuid);
}

Texture2D *World::getTexture2DByGuid(const Guid &assetGuid) const
{
    return getAssetByGuid<Texture2D>(assetGuid);
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

Guid World::getGuidFromId(const Id &id) const
{
    std::unordered_map<Id, Guid>::const_iterator it = mIdState.mIdToGuid.find(id);
    if (it != mIdState.mIdToGuid.end())
    {
        return it->second;
    }

    return Guid::INVALID;
}

Id World::getIdFromGuid(const Guid &guid) const
{
    std::unordered_map<Guid, Id>::const_iterator it = mIdState.mGuidToId.find(guid);
    if (it != mIdState.mGuidToId.end())
    {
        return it->second;
    }

    return Id::INVALID;
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

Cubemap *World::createCubemap(const YAML::Node &in)
{
    return createAsset<Cubemap>(in);
}

Material *World::createMaterial(const YAML::Node &in)
{
    return createAsset<Material>(in);
}

Mesh *World::createMesh(const YAML::Node &in)
{
    return createAsset<Mesh>(in);
}

RenderTexture *World::createRenderTexture(const YAML::Node &in)
{
    return createAsset<RenderTexture>(in);
}

Shader *World::createShader(const YAML::Node &in)
{
    return createAsset<Shader>(in);
}

Texture2D *World::createTexture2D(const YAML::Node &in)
{
    return createAsset<Texture2D>(in);
}

void World::latentDestroyAsset(const Guid &assetGuid, int assetType)
{
    mIdState.mAssetGuidsMarkedLatentDestroy.push_back(std::make_pair(assetGuid, assetType));
}

void World::immediateDestroyAsset(const Guid &assetGuid, int assetType)
{
    int index = getIndexOf(assetGuid);
    Id assetId = getIdFromGuid(assetGuid);

    switch (assetType)
    {
    case AssetType<Cubemap>::type: {
        Cubemap *swap = mAllocators.mCubemapAllocator.destruct(index);

        removeFromIdState_impl<Cubemap>(assetGuid, assetId);

        if (swap != nullptr)
        {
            addToIdState_impl<Cubemap>(swap->getGuid(), swap->getId(), index, assetType);
        }
        break;
    }
    case AssetType<Material>::type: {
        Material *swap = mAllocators.mMaterialAllocator.destruct(index);

        removeFromIdState_impl<Material>(assetGuid, assetId);

        if (swap != nullptr)
        {
            addToIdState_impl<Material>(swap->getGuid(), swap->getId(), index, assetType);
        }
        break;
    }
    case AssetType<Mesh>::type: {
        Mesh *swap = mAllocators.mMeshAllocator.destruct(index);

        removeFromIdState_impl<Mesh>(assetGuid, assetId);

        if (swap != nullptr)
        {
            addToIdState_impl<Mesh>(swap->getGuid(), swap->getId(), index, assetType);
        }
        break;
    }
    case AssetType<Shader>::type: {
        Shader *swap = mAllocators.mShaderAllocator.destruct(index);

        removeFromIdState_impl<Shader>(assetGuid, assetId);

        if (swap != nullptr)
        {
            addToIdState_impl<Shader>(swap->getGuid(), swap->getId(), index, assetType);
        }
        break;
    }
    case AssetType<Texture2D>::type: {
        Texture2D *swap = mAllocators.mTexture2DAllocator.destruct(index);

        removeFromIdState_impl<Texture2D>(assetGuid, assetId);

        if (swap != nullptr)
        {
            addToIdState_impl<Texture2D>(swap->getGuid(), swap->getId(), index, assetType);
        }
        break;
    }
    case AssetType<RenderTexture>::type: {
        RenderTexture *swap = mAllocators.mRenderTextureAllocator.destruct(index);

        removeFromIdState_impl<RenderTexture>(assetGuid, assetId);

        if (swap != nullptr)
        {
            addToIdState_impl<RenderTexture>(swap->getGuid(), swap->getId(), index, assetType);
        }
        break;
    }
    default:
        assert(!"Unreachable code");
    }
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