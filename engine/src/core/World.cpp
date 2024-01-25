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

    T *oldComponent = from->getComponentByGuid<T>(guid);

    assert(oldComponent != nullptr);

    YAML::Node oldComponentNode;
    oldComponent->serialize(oldComponentNode);

    if constexpr(ComponentType<T>::type == ComponentType<Transform>::type)
    {
        TransformData *transformData = from->getTransformDataFromTransformGuid(guid);
        transformData->serialize(oldComponentNode);
    }

    T *newComponent = to->getComponentByGuid<T>(guid);

    if (newComponent == nullptr)
    {
        to->addComponent<T>(oldComponentNode);   
    }
    else
    {
        newComponent->deserialize(oldComponentNode);
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
    std::unordered_map<Id, int>::const_iterator it = mIdState.mMeshIdToGlobalIndex.find(assetId);
    return (it != mIdState.mMeshIdToGlobalIndex.end()) ? mAllocators.mMeshAllocator.get(it->second) : nullptr;
}

template <> Material *World::getAssetById<Material>(const Id &assetId) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mMaterialIdToGlobalIndex.find(assetId);
    return (it != mIdState.mMaterialIdToGlobalIndex.end()) ? mAllocators.mMaterialAllocator.get(it->second) : nullptr;
}

template <> Shader *World::getAssetById<Shader>(const Id &assetId) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mShaderIdToGlobalIndex.find(assetId);
    return (it != mIdState.mShaderIdToGlobalIndex.end()) ? mAllocators.mShaderAllocator.get(it->second) : nullptr;
}

template <> Texture2D *World::getAssetById<Texture2D>(const Id &assetId) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mTexture2DIdToGlobalIndex.find(assetId);
    return (it != mIdState.mTexture2DIdToGlobalIndex.end()) ? mAllocators.mTexture2DAllocator.get(it->second) : nullptr;
}

template <> Cubemap *World::getAssetById<Cubemap>(const Id &assetId) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mCubemapIdToGlobalIndex.find(assetId);
    return (it != mIdState.mCubemapIdToGlobalIndex.end()) ? mAllocators.mCubemapAllocator.get(it->second) : nullptr;
}

template <> RenderTexture *World::getAssetById<RenderTexture>(const Id &assetId) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mRenderTextureIdToGlobalIndex.find(assetId);
    return (it != mIdState.mRenderTextureIdToGlobalIndex.end()) ? mAllocators.mRenderTextureAllocator.get(it->second) : nullptr;
}

template <> Mesh *World::getAssetByGuid<Mesh>(const Guid &assetGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mMeshGuidToGlobalIndex.find(assetGuid);
    return (it != mIdState.mMeshGuidToGlobalIndex.end()) ? mAllocators.mMeshAllocator.get(it->second) : nullptr;
}

template <> Material *World::getAssetByGuid<Material>(const Guid &assetGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mMaterialGuidToGlobalIndex.find(assetGuid);
    return (it != mIdState.mMaterialGuidToGlobalIndex.end()) ? mAllocators.mMaterialAllocator.get(it->second) : nullptr;
}

template <> Shader *World::getAssetByGuid<Shader>(const Guid &assetGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mShaderGuidToGlobalIndex.find(assetGuid);
    return (it != mIdState.mShaderGuidToGlobalIndex.end()) ? mAllocators.mShaderAllocator.get(it->second) : nullptr;
}

template <> Texture2D *World::getAssetByGuid<Texture2D>(const Guid &assetGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mTexture2DGuidToGlobalIndex.find(assetGuid);
    return (it != mIdState.mTexture2DGuidToGlobalIndex.end()) ? mAllocators.mTexture2DAllocator.get(it->second) : nullptr;
}

template <> Cubemap *World::getAssetByGuid<Cubemap>(const Guid &assetGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mCubemapGuidToGlobalIndex.find(assetGuid);
    return (it != mIdState.mCubemapGuidToGlobalIndex.end()) ? mAllocators.mCubemapAllocator.get(it->second) : nullptr;
}

template <> RenderTexture *World::getAssetByGuid<RenderTexture>(const Guid &assetGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mRenderTextureGuidToGlobalIndex.find(assetGuid);
    return (it != mIdState.mRenderTextureGuidToGlobalIndex.end()) ? mAllocators.mRenderTextureAllocator.get(it->second) : nullptr;
}

template <> Mesh *World::createAsset<Mesh>()
{
    return addMesh(Guid::newGuid());
}

template <> Material *World::createAsset<Material>()
{
    return addMaterial(Guid::newGuid());
}

template <> Shader *World::createAsset<Shader>()
{
    return addShader(Guid::newGuid());
}

template <> Texture2D *World::createAsset<Texture2D>()
{
    return addTexture2D(Guid::newGuid());
}

template <> Cubemap *World::createAsset<Cubemap>()
{
    return addCubemap(Guid::newGuid());
}

template <> RenderTexture *World::createAsset<RenderTexture>()
{
    return addRenderTexture(Guid::newGuid());
}

template <> Mesh *World::createAsset<Mesh>(const Guid &assetGuid)
{
    return addMesh(assetGuid);
}

template <> Material *World::createAsset<Material>(const Guid &assetGuid)
{
    return addMaterial(assetGuid);
}

template <> Shader *World::createAsset<Shader>(const Guid &assetGuid)
{
    return addShader(assetGuid);
}

template <> Texture2D *World::createAsset<Texture2D>(const Guid &assetGuid)
{
    return addTexture2D(assetGuid);
}

template <> Cubemap *World::createAsset<Cubemap>(const Guid &assetGuid)
{
    return addCubemap(assetGuid);
}

template <> RenderTexture *World::createAsset<RenderTexture>(const Guid &assetGuid)
{
    return addRenderTexture(assetGuid);
}

template <> Mesh *World::createAsset<Mesh>(const YAML::Node &in)
{
    return addMesh(in);
}

template <> Material *World::createAsset<Material>(const YAML::Node &in)
{
    return addMaterial(in);
}

template <> Shader *World::createAsset<Shader>(const YAML::Node &in)
{
    return addShader(in);
}

template <> Texture2D *World::createAsset<Texture2D>(const YAML::Node &in)
{
    return addTexture2D(in);
}

template <> Cubemap *World::createAsset<Cubemap>(const YAML::Node &in)
{
    return addCubemap(in);
}

template <> RenderTexture *World::createAsset<RenderTexture>(const YAML::Node &in)
{
    return addRenderTexture(in);
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
    mPrimitives.createPrimitiveMeshes(this, 100, 100);
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

//Material *loadMaterialFromYAML(const std::string &filePath);
//Mesh *loadMeshFromYAML(const std::string &filePath);
//RenderTexture *loadRenderTextureFromYAML(const std::string &filePath);
//Texture2D *loadTexture2DFromYAML(const std::string &filePath);

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
    std::unordered_map<Id, int>::const_iterator it = mIdState.mSceneIdToGlobalIndex.find(sceneId);
    return (it != mIdState.mSceneIdToGlobalIndex.end()) ? mAllocators.mSceneAllocator.get(it->second)
                                                                  : nullptr;
}

Scene *World::getSceneByGuid(const Guid &sceneGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mSceneGuidToGlobalIndex.find(sceneGuid);
    return (it != mIdState.mSceneGuidToGlobalIndex.end()) ? mAllocators.mSceneAllocator.get(it->second)
                                                                  : nullptr;
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
    return addScene();
}

Scene *World::createScene(const YAML::Node &in)
{
    return addScene(in);
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
    switch (assetType)
    {
    case AssetType<Cubemap>::type: {
        removeCubemap(assetGuid);
        break;
    }
    case AssetType<Material>::type: {
        removeMaterial(assetGuid);
        break;
    }
    case AssetType<Mesh>::type: {
        removeMesh(assetGuid);
        break;
    }
    case AssetType<Shader>::type: {
        removeShader(assetGuid);
        break;
    }
    case AssetType<Texture2D>::type: {
        removeTexture2D(assetGuid);
        break;
    }
    case AssetType<RenderTexture>::type: {
        removeRenderTexture(assetGuid);
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

void World::addToIdState(std::unordered_map<Guid, int> &guidToIndex, std::unordered_map<Id, int> &idToIndex,
                              int index, const Guid &guid, const Id &id, int type)
{
    assert(guid != Guid::INVALID);
    assert(id != Id::INVALID);
    assert(index >= 0);

    guidToIndex[guid] = index;
    idToIndex[id] = index;

    mIdState.mGuidToGlobalIndex[guid] = index;
    mIdState.mIdToGlobalIndex[id] = index;

    mIdState.mGuidToType[guid] = type;
    mIdState.mIdToType[id] = type;

    mIdState.mGuidToId[guid] = id;
    mIdState.mIdToGuid[id] = guid;
}

void World::removeFromIdState(std::unordered_map<Guid, int> &guidToIndex, std::unordered_map<Id, int> &idToIndex,
                                   const Guid &guid, const Id &id)
{
    assert(guid != Guid::INVALID);
    assert(id != Id::INVALID);

    guidToIndex.erase(guid);
    idToIndex.erase(id);

    mIdState.mGuidToGlobalIndex.erase(guid);
    mIdState.mIdToGlobalIndex.erase(id);

    mIdState.mGuidToType.erase(guid);
    mIdState.mIdToType.erase(id);

    mIdState.mGuidToId.erase(guid);
    mIdState.mIdToGuid.erase(id);
}

void World::moveIndexInIdState(std::unordered_map<Guid, int> &guidToIndex, std::unordered_map<Id, int> &idToIndex,
                             const Guid &guid, const Id &id, int index)
{
    assert(guid != Guid::INVALID);
    assert(id != Id::INVALID);
    assert(index >= 0);

    guidToIndex[guid] = index;
    idToIndex[id] = index;

    mIdState.mGuidToGlobalIndex[guid] = index;
    mIdState.mIdToGlobalIndex[id] = index;
}

Scene *World::addScene(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mSceneAllocator.getCount();

    Scene *scene = mAllocators.mSceneAllocator.construct(this, Id::newId());
    scene->deserialize(in);

    addToIdState(mIdState.mSceneGuidToGlobalIndex, mIdState.mSceneIdToGlobalIndex, globalIndex,
                      scene->getGuid(), scene->getId(), AssetType<Scene>::type);

    return scene;
}

Cubemap *World::addCubemap(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mCubemapAllocator.getCount();

    Cubemap *cubemap = mAllocators.mCubemapAllocator.construct(this, Id::newId());
    cubemap->deserialize(in);

    addToIdState(mIdState.mCubemapGuidToGlobalIndex, mIdState.mCubemapIdToGlobalIndex, globalIndex,
                      cubemap->getGuid(), cubemap->getId(), AssetType<Cubemap>::type);
    return cubemap;
}

Material *World::addMaterial(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mMaterialAllocator.getCount();

    Material *material = mAllocators.mMaterialAllocator.construct(this, Id::newId());
    material->deserialize(in);

    addToIdState(mIdState.mMaterialGuidToGlobalIndex, mIdState.mMaterialIdToGlobalIndex, globalIndex,
                      material->getGuid(), material->getId(), AssetType<Material>::type);
    return material;
}

Mesh *World::addMesh(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mMeshAllocator.getCount();

    Mesh *mesh = mAllocators.mMeshAllocator.construct(this, Id::newId());
    mesh->deserialize(in);

    addToIdState(mIdState.mMeshGuidToGlobalIndex, mIdState.mMeshIdToGlobalIndex, globalIndex,
                      mesh->getGuid(), mesh->getId(), AssetType<Mesh>::type);
    return mesh;
}

Shader *World::addShader(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mShaderAllocator.getCount();

    Shader *shader = mAllocators.mShaderAllocator.construct(this, Id::newId());
    shader->deserialize(in);

    addToIdState(mIdState.mShaderGuidToGlobalIndex, mIdState.mShaderIdToGlobalIndex, globalIndex,
                      shader->getGuid(), shader->getId(), AssetType<Shader>::type);
    return shader;
}

Texture2D *World::addTexture2D(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mTexture2DAllocator.getCount();

    Texture2D *texture = mAllocators.mTexture2DAllocator.construct(this, Id::newId());
    texture->deserialize(in);

    addToIdState(mIdState.mTexture2DGuidToGlobalIndex, mIdState.mTexture2DIdToGlobalIndex, globalIndex,
                      texture->getGuid(), texture->getId(), AssetType<Texture2D>::type);
    return texture;
}

RenderTexture *World::addRenderTexture(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mRenderTextureAllocator.getCount();

    RenderTexture *texture = mAllocators.mRenderTextureAllocator.construct(this, Id::newId());
    texture->deserialize(in);

    addToIdState(mIdState.mRenderTextureGuidToGlobalIndex, mIdState.mRenderTextureIdToGlobalIndex, globalIndex,
                      texture->getGuid(), texture->getId(), AssetType<RenderTexture>::type);
    return texture;
}


Scene *World::addScene()
{
    int globalIndex = (int)mAllocators.mSceneAllocator.getCount();

    Scene *scene = mAllocators.mSceneAllocator.construct(this, Guid::newGuid(), Id::newId());

    addToIdState(mIdState.mSceneGuidToGlobalIndex, mIdState.mSceneIdToGlobalIndex, globalIndex, scene->getGuid(),
                      scene->getId(), AssetType<Scene>::type);

    return scene;
}

Cubemap *World::addCubemap(const Guid &assetGuid)
{
    int globalIndex = (int)mAllocators.mCubemapAllocator.getCount();

    Cubemap *cubemap = mAllocators.mCubemapAllocator.construct(this, assetGuid, Id::newId());
    
    addToIdState(mIdState.mCubemapGuidToGlobalIndex, mIdState.mCubemapIdToGlobalIndex, globalIndex,
                      cubemap->getGuid(), cubemap->getId(), AssetType<Cubemap>::type);
    return cubemap;
}

Material *World::addMaterial(const Guid &assetGuid)
{
    int globalIndex = (int)mAllocators.mMaterialAllocator.getCount();

    Material *material = mAllocators.mMaterialAllocator.construct(this, assetGuid, Id::newId());

    addToIdState(mIdState.mMaterialGuidToGlobalIndex, mIdState.mMaterialIdToGlobalIndex, globalIndex,
                      material->getGuid(), material->getId(), AssetType<Material>::type);
    return material;
}

Mesh *World::addMesh(const Guid &assetGuid)
{
    int globalIndex = (int)mAllocators.mMeshAllocator.getCount();

    Mesh *mesh = mAllocators.mMeshAllocator.construct(this, assetGuid, Id::newId());

    addToIdState(mIdState.mMeshGuidToGlobalIndex, mIdState.mMeshIdToGlobalIndex, globalIndex,
                      mesh->getGuid(), mesh->getId(), AssetType<Mesh>::type);
    return mesh;
}

Shader *World::addShader(const Guid &assetGuid)
{
    int globalIndex = (int)mAllocators.mShaderAllocator.getCount();

    Shader *shader = mAllocators.mShaderAllocator.construct(this, assetGuid, Id::newId());

    addToIdState(mIdState.mShaderGuidToGlobalIndex, mIdState.mShaderIdToGlobalIndex, globalIndex,
                      shader->getGuid(), shader->getId(), AssetType<Shader>::type);
    return shader;
}

Texture2D *World::addTexture2D(const Guid &assetGuid)
{
    int globalIndex = (int)mAllocators.mTexture2DAllocator.getCount();

    Texture2D *texture = mAllocators.mTexture2DAllocator.construct(this, assetGuid, Id::newId());

    addToIdState(mIdState.mTexture2DGuidToGlobalIndex, mIdState.mTexture2DIdToGlobalIndex, globalIndex,
                      texture->getGuid(), texture->getId(), AssetType<Texture2D>::type);
    return texture;
}

RenderTexture *World::addRenderTexture(const Guid &assetGuid)
{
    int globalIndex = (int)mAllocators.mRenderTextureAllocator.getCount();

    RenderTexture *texture = mAllocators.mRenderTextureAllocator.construct(this, assetGuid, Id::newId());

    addToIdState(mIdState.mRenderTextureGuidToGlobalIndex, mIdState.mRenderTextureIdToGlobalIndex, globalIndex,
                      texture->getGuid(), texture->getId(), AssetType<RenderTexture>::type);
    return texture;
}

void World::removeCubemap(const Guid &assetGuid)
{
    int assetIndex = getIndexOf(assetGuid);
    Id assetId = getIdFromGuid(assetGuid);

    Cubemap *swap = mAllocators.mCubemapAllocator.destruct(assetIndex);

    removeFromIdState(mIdState.mCubemapGuidToGlobalIndex, mIdState.mCubemapIdToGlobalIndex,
                           assetGuid, assetId);

    if (swap != nullptr)
    {
        moveIndexInIdState(mIdState.mCubemapGuidToGlobalIndex, mIdState.mCubemapIdToGlobalIndex,
                                    assetGuid, assetId, assetIndex);
    }
}

void World::removeMaterial(const Guid &assetGuid)
{
    int assetIndex = getIndexOf(assetGuid);
    Id assetId = getIdFromGuid(assetGuid);

    Material *swap = mAllocators.mMaterialAllocator.destruct(assetIndex);

    removeFromIdState(mIdState.mMaterialGuidToGlobalIndex, mIdState.mMaterialIdToGlobalIndex, assetGuid, assetId);

    if (swap != nullptr)
    {
        moveIndexInIdState(mIdState.mMaterialGuidToGlobalIndex, mIdState.mMaterialIdToGlobalIndex, assetGuid,
                                assetId, assetIndex);
    }
}

void World::removeMesh(const Guid &assetGuid)
{
    int assetIndex = getIndexOf(assetGuid);
    Id assetId = getIdFromGuid(assetGuid);

    Mesh *swap = mAllocators.mMeshAllocator.destruct(assetIndex);

    removeFromIdState(mIdState.mMeshGuidToGlobalIndex, mIdState.mMeshIdToGlobalIndex, assetGuid, assetId);

    if (swap != nullptr)
    {
        moveIndexInIdState(mIdState.mMeshGuidToGlobalIndex, mIdState.mMeshIdToGlobalIndex, assetGuid,
                                assetId, assetIndex);
    }
}

void World::removeShader(const Guid &assetGuid)
{
    int assetIndex = getIndexOf(assetGuid);
    Id assetId = getIdFromGuid(assetGuid);

    Shader *swap = mAllocators.mShaderAllocator.destruct(assetIndex);

    removeFromIdState(mIdState.mShaderGuidToGlobalIndex, mIdState.mShaderIdToGlobalIndex, assetGuid, assetId);

    if (swap != nullptr)
    {
        moveIndexInIdState(mIdState.mShaderGuidToGlobalIndex, mIdState.mShaderIdToGlobalIndex, assetGuid,
                                assetId, assetIndex);
    }
}

void World::removeTexture2D(const Guid &assetGuid)
{
    int assetIndex = getIndexOf(assetGuid);
    Id assetId = getIdFromGuid(assetGuid);

    Texture2D *swap = mAllocators.mTexture2DAllocator.destruct(assetIndex);

    removeFromIdState(mIdState.mTexture2DGuidToGlobalIndex, mIdState.mTexture2DIdToGlobalIndex, assetGuid, assetId);

    if (swap != nullptr)
    {
        moveIndexInIdState(mIdState.mTexture2DGuidToGlobalIndex, mIdState.mTexture2DIdToGlobalIndex, assetGuid,
                                assetId, assetIndex);
    }
}

void World::removeRenderTexture(const Guid &assetGuid)
{
    int assetIndex = getIndexOf(assetGuid);
    Id assetId = getIdFromGuid(assetGuid);

    RenderTexture *swap = mAllocators.mRenderTextureAllocator.destruct(assetIndex);

    removeFromIdState(mIdState.mRenderTextureGuidToGlobalIndex, mIdState.mRenderTextureIdToGlobalIndex, assetGuid,
                           assetId);

    if (swap != nullptr)
    {
        moveIndexInIdState(mIdState.mRenderTextureGuidToGlobalIndex, mIdState.mRenderTextureIdToGlobalIndex,
                                assetGuid,
                                assetId, assetIndex);
    }
}