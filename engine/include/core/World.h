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
    // allocators for assets, entities, components, and systems
    WorldAllocators mAllocators;

    // id state for assets, entities, components, and systems
    WorldIdState mIdState;

    // Primitive meshes all worlds have access to
    WorldPrimitives mPrimitives;

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

    void latentDestroyEntitiesInWorld();
    void immediateDestroyEntitiesInWorld();

    std::vector<ShaderUniform> getCachedMaterialUniforms(const Guid &materialId, const Guid &shaderId);
    void cacheMaterialUniforms(const Guid &materialId, const Guid &shaderId, const std::vector<ShaderUniform>& uniforms);

    size_t getNumberOfScenes() const;
    size_t getNumberOfEntities() const;
    size_t getNumberOfNonHiddenEntities() const;
    size_t getNumberOfUpdatingSystems() const;
    Mesh *getPrimtiveMesh(PrimitiveType type) const;
    Entity *createPrimitive(PrimitiveType type);
    Entity *createNonPrimitive(const Guid &meshId);
    Entity *createLight(LightType type);
    Entity *createCamera();

    template <typename T> size_t getNumberOfSystems() const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return getNumberOfSystems_impl<T>(getSystemAllocator_impl<T>());
    }

    template <typename T> size_t getNumberOfComponents() const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return getNumberOfComponents_impl<T>(getComponentAllocator_impl<T>());
    }

    template <typename T> size_t getNumberOfAssets() const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return getNumberOfAssets_impl(getAssetAllocator_impl<T>());
    }

    template <typename T> T *getSystem() const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return getSystem_impl(getSystemAllocator_impl<T>());
    }

    template <typename T> T *getComponent(const Guid &entityId) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return getComponent_impl<T>(getComponentAllocator_impl<T>(), entityId);
    }

    template <typename T> T *addComponent(const Guid &entityId)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");
        static_assert(!std::is_base_of<Transform, T>(), "'T' cannot be of type Transform");

        return addComponent_impl<T>(getComponentOrAddAllocator_impl<T>(), entityId);
    }

    template <typename T> T *addSystem(size_t order)
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return addSystem_impl<T>(getSystemOrAddAllocator_impl<T>(), order);
    }

    Scene *getSceneByIndex(size_t index) const;
    Scene *getSceneById(const Guid &sceneId) const;
    Entity *getEntityByIndex(size_t index) const;
    Entity *getEntityById(const Guid &entityId) const;
    System *getSystemByUpdateOrder(size_t order) const;

    template <typename T> T *getSystemByIndex(size_t index) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return getSystemByIndex_impl(getSystemAllocator_impl<T>(), index);
    }

    template <typename T> T *getSystemById(const Guid &systemId) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return getSystemById_impl<T>(getSystemAllocator_impl<T>(), systemId);
    }

    template <typename T> T *getAssetByIndex(size_t index) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return getAssetByIndex_impl(getAssetAllocator_impl<T>(), index);
    }

    template <typename T> T *getAssetById(const Guid &assetId) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return getAssetById_impl<T>(getAssetAllocator_impl<T>(), assetId);
    }

    template <typename T> T *getComponentByIndex(size_t index) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return getComponentByIndex_impl(getComponentAllocator_impl<T>(), index);
    }

    template <typename T> T *getComponentById(const Guid &componentId) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return getComponentById_impl<T>(getComponentAllocator_impl<T>(), componentId);
    }

    template <typename T> T *createAsset()
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return createAsset_impl<T>(getAssetOrAddAllocator_impl<T>(), Guid::newGuid());
    }

    template <typename T> T *createAsset(const Guid& id)
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return createAsset_impl<T>(getAssetOrAddAllocator_impl<T>(), id);
    }

    int getIndexOf(const Guid &id) const;
    int getTypeOf(const Guid &id) const;

    Scene *createScene();
    Entity *createEntity();

    void latentDestroyEntity(const Guid &entityId);
    void immediateDestroyEntity(const Guid &entityId);
    void latentDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType);
    void immediateDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType);
    void latentDestroyAsset(const Guid &assetId, int assetType);
    void immediateDestroyAsset(const Guid &assetId, int assetType);

    bool isMarkedForLatentDestroy(const Guid &id);
    void clearIdsMarkedCreatedOrDestroyed();

    std::vector<std::pair<Guid, int>> getComponentsOnEntity(const Guid &entityId) const;

    std::vector<Guid> getEntityIdsMarkedCreated() const;
    std::vector<Guid> getEntityIdsMarkedLatentDestroy() const;
    std::vector<std::tuple<Guid, Guid, int>> getComponentIdsMarkedCreated() const;
    std::vector<std::tuple<Guid, Guid, int>> getComponentIdsMarkedLatentDestroy() const;

    std::string getAssetFilepath(const Guid &assetId) const;
    std::string getSceneFilepath(const Guid &sceneId) const;

    Guid getAssetId(const std::string& filepath) const;
    Guid getSceneId(const std::string& filepath) const;

  private:
    Asset *loadAssetFromYAML_impl(const YAML::Node &in);
    Scene *loadSceneFromYAML_impl(const YAML::Node &in);

    void generateSourcePaths(const std::string& filepath, YAML::Node &in);

    template <typename T> size_t getNumberOfSystems_impl(const PoolAllocator<T> *allocator) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return allocator != nullptr ? allocator->getCount() : 0;
    }

    template <typename T> size_t getNumberOfComponents_impl(const PoolAllocator<T> *allocator) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

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

    template <typename T> T *getComponent_impl(const PoolAllocator<T> *allocator, const Guid &entityId) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        assert(allocator != nullptr);

        std::vector<std::pair<Guid, int>> componentsOnEntity = getComponentsOnEntity(entityId);

        for (size_t i = 0; i < componentsOnEntity.size(); i++)
        {
            if (ComponentType<T>::type == componentsOnEntity[i].second)
            {
                std::unordered_map<Guid, int>::const_iterator it =
                    mIdState.mIdToGlobalIndex.find(componentsOnEntity[i].first);
                if (it != mIdState.mIdToGlobalIndex.end())
                {
                    return allocator->get(it->second);
                }

                break;
            }
        }

        return nullptr;
    }

    template <typename T> T *addComponent_impl(PoolAllocator<T> *allocator, const Guid &entityId)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        assert(allocator != nullptr);

        if (getTypeOf(entityId) != EntityType<Entity>::type)
        {
            return nullptr;
        }

        int componentGlobalIndex = (int)allocator->getCount();
        int componentType = ComponentType<T>::type;
        Guid componentId = Guid::newGuid();

        T *component = allocator->construct(this, componentId);

        if (component != nullptr)
        {
            component->mEntityId = entityId;

            addIdToGlobalIndexMap_impl<T>(componentId, componentGlobalIndex, componentType);

            mIdState.mEntityIdToComponentIds[entityId].push_back(std::make_pair(componentId, componentType));

            mIdState.mComponentIdsMarkedCreated.push_back(std::make_tuple(entityId, componentId, componentType));
        }

        return component;
    }

    template <typename T> T *addSystem_impl(PoolAllocator<T> *allocator, size_t order)
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        int systemGlobalIndex = (int)allocator->getCount();
        int systemType = SystemType<T>::type;
        Guid systemId = Guid::newGuid();

        T *system = allocator->construct(this, systemId);

        if (system != nullptr)
        {
            addIdToGlobalIndexMap_impl<T>(systemId, systemGlobalIndex, systemType);

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

    template <typename T> T *getSystemById_impl(const PoolAllocator<T> *allocator, const Guid &systemId) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        if (allocator == nullptr || SystemType<T>::type != getTypeOf(systemId))
        {
            return nullptr;
        }

        return getById_impl<T>(mIdState.mIdToGlobalIndex, allocator, systemId);
    }

    template <typename T> T *getAssetByIndex_impl(const PoolAllocator<T> *allocator, size_t index) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return allocator != nullptr ? allocator->get(index) : nullptr;
    }

    template <typename T> T *getAssetById_impl(const PoolAllocator<T> *allocator, const Guid &assetId) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        if (allocator == nullptr || AssetType<T>::type != getTypeOf(assetId))
        {
            return nullptr;
        }

        return getById_impl<T>(mIdState.mIdToGlobalIndex, allocator, assetId);
    }

    template <typename T> T *getComponentByIndex_impl(const PoolAllocator<T> *allocator, size_t index) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return allocator != nullptr ? allocator->get(index) : nullptr;
    }

    template <typename T> T *getComponentById_impl(const PoolAllocator<T> *allocator, const Guid &componentId) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        if (allocator == nullptr || ComponentType<T>::type != getTypeOf(componentId))
        {
            return nullptr;
        }

        return getById_impl<T>(mIdState.mIdToGlobalIndex, allocator, componentId);
    }

    template <typename T> T *createAsset_impl(PoolAllocator<T> *allocator, const Guid& id)
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        int index = (int)allocator->getCount();
        int type = AssetType<T>::type;

        T *asset = allocator->construct(this, id);

        if (asset != nullptr)
        {
            addIdToGlobalIndexMap_impl<T>(id, index, type);
        }

        return asset;
    }

    template <typename T> PoolAllocator<T> *getComponentAllocator_impl() const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        std::unordered_map<int, Allocator *>::const_iterator it =
            mAllocators.mComponentAllocatorMap.find(ComponentType<T>::type);
        if (it != mAllocators.mComponentAllocatorMap.end())
        {
            return static_cast<PoolAllocator<T> *>(it->second);
        }

        return nullptr;
    }

    template <typename T> PoolAllocator<T> *getComponentOrAddAllocator_impl()
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        PoolAllocator<T> *allocator = getComponentAllocator_impl<T>();
        if (allocator == nullptr)
        {
            allocator = new PoolAllocator<T>();
            mAllocators.mComponentAllocatorMap[ComponentType<T>::type] = allocator;
        }

        return allocator;
    }

    template <typename T> PoolAllocator<T> *getSystemAllocator_impl() const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        std::unordered_map<int, Allocator *>::const_iterator it =
            mAllocators.mSystemAllocatorMap.find(SystemType<T>::type);
        if (it != mAllocators.mSystemAllocatorMap.end())
        {
            return static_cast<PoolAllocator<T> *>(it->second);
        }

        return nullptr;
    }

    template <typename T> PoolAllocator<T> *getSystemOrAddAllocator_impl()
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        PoolAllocator<T> *allocator = getSystemAllocator_impl<T>();
        if (allocator == nullptr)
        {
            allocator = new PoolAllocator<T>();
            mAllocators.mSystemAllocatorMap[SystemType<T>::type] = allocator;
        }

        return allocator;
    }

    template <typename T> PoolAllocator<T> *getAssetAllocator_impl() const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        std::unordered_map<int, Allocator *>::const_iterator it =
            mAllocators.mAssetAllocatorMap.find(AssetType<T>::type);
        if (it != mAllocators.mAssetAllocatorMap.end())
        {
            return static_cast<PoolAllocator<T> *>(it->second);
        }

        return nullptr;
    }

    template <typename T> PoolAllocator<T> *getAssetOrAddAllocator_impl()
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        PoolAllocator<T> *allocator = getAssetAllocator_impl<T>();
        if (allocator == nullptr)
        {
            allocator = new PoolAllocator<T>();
            mAllocators.mAssetAllocatorMap[AssetType<T>::type] = allocator;
        }

        return allocator;
    }

    template <typename T>
    T *getById_impl(const std::unordered_map<Guid, int> &idToIndexMap, const PoolAllocator<T> *allocator,
                    const Guid &id) const
    {
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
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <int N> void removeIdToGlobalIndexMap_impl(const Guid &id, int type)
    {
        mIdState.mIdToGlobalIndex.erase(id);
        mIdState.mIdToType.erase(id);
    }
};

// Explicit template specializations

template <> size_t World::getNumberOfSystems<RenderSystem>() const;
template <> size_t World::getNumberOfSystems<PhysicsSystem>() const;
template <> size_t World::getNumberOfSystems<CleanUpSystem>() const;
template <> size_t World::getNumberOfSystems<DebugSystem>() const;
template <> size_t World::getNumberOfSystems<GizmoSystem>() const;
template <> size_t World::getNumberOfSystems<FreeLookCameraSystem>() const;
template <> size_t World::getNumberOfSystems<TerrainSystem>() const;
template <> size_t World::getNumberOfComponents<Transform>() const;
template <> size_t World::getNumberOfComponents<MeshRenderer>() const;
template <> size_t World::getNumberOfComponents<SpriteRenderer>() const;
template <> size_t World::getNumberOfComponents<LineRenderer>() const;
template <> size_t World::getNumberOfComponents<Rigidbody>() const;
template <> size_t World::getNumberOfComponents<Camera>() const;
template <> size_t World::getNumberOfComponents<Light>() const;
template <> size_t World::getNumberOfComponents<SphereCollider>() const;
template <> size_t World::getNumberOfComponents<BoxCollider>() const;
template <> size_t World::getNumberOfComponents<CapsuleCollider>() const;
template <> size_t World::getNumberOfComponents<MeshCollider>() const;
template <> size_t World::getNumberOfComponents<Terrain>() const;
template <> size_t World::getNumberOfAssets<Mesh>() const;
template <> size_t World::getNumberOfAssets<Material>() const;
template <> size_t World::getNumberOfAssets<Shader>() const;
template <> size_t World::getNumberOfAssets<Texture2D>() const;
template <> size_t World::getNumberOfAssets<Texture3D>() const;
template <> size_t World::getNumberOfAssets<Cubemap>() const;
template <> size_t World::getNumberOfAssets<RenderTexture>() const;
template <> size_t World::getNumberOfAssets<Font>() const;
template <> size_t World::getNumberOfAssets<Sprite>() const;
template <> RenderSystem *World::getSystem<RenderSystem>() const;
template <> PhysicsSystem *World::getSystem<PhysicsSystem>() const;
template <> CleanUpSystem *World::getSystem<CleanUpSystem>() const;
template <> DebugSystem *World::getSystem<DebugSystem>() const;
template <> GizmoSystem *World::getSystem<GizmoSystem>() const;
template <> FreeLookCameraSystem *World::getSystem<FreeLookCameraSystem>() const;
template <> TerrainSystem *World::getSystem<TerrainSystem>() const;
template <> Transform *World::getComponent<Transform>(const Guid &entityId) const;
template <> MeshRenderer *World::getComponent<MeshRenderer>(const Guid &entityId) const;
template <> SpriteRenderer *World::getComponent<SpriteRenderer>(const Guid &entityId) const;
template <> LineRenderer *World::getComponent<LineRenderer>(const Guid &entityId) const;
template <> Rigidbody *World::getComponent<Rigidbody>(const Guid &entityId) const;
template <> Camera *World::getComponent<Camera>(const Guid &entityId) const;
template <> Light *World::getComponent<Light>(const Guid &entityId) const;
template <> SphereCollider *World::getComponent<SphereCollider>(const Guid &entityId) const;
template <> BoxCollider *World::getComponent<BoxCollider>(const Guid &entityId) const;
template <> CapsuleCollider *World::getComponent<CapsuleCollider>(const Guid &entityId) const;
template <> MeshCollider *World::getComponent<MeshCollider>(const Guid &entityId) const;
template <> Terrain *World::getComponent<Terrain>(const Guid &entityId) const;
template <> MeshRenderer *World::addComponent<MeshRenderer>(const Guid &entityId);
template <> SpriteRenderer *World::addComponent<SpriteRenderer>(const Guid &entityId);
template <> LineRenderer *World::addComponent<LineRenderer>(const Guid &entityId);
template <> Rigidbody *World::addComponent<Rigidbody>(const Guid &entityId);
template <> Camera *World::addComponent<Camera>(const Guid &entityId);
template <> Light *World::addComponent<Light>(const Guid &entityId);
template <> SphereCollider *World::addComponent<SphereCollider>(const Guid &entityId);
template <> BoxCollider *World::addComponent<BoxCollider>(const Guid &entityId);
template <> CapsuleCollider *World::addComponent<CapsuleCollider>(const Guid &entityId);
template <> MeshCollider *World::addComponent<MeshCollider>(const Guid &entityId);
template <> Terrain *World::addComponent<Terrain>(const Guid &entityId);
template <> RenderSystem *World::addSystem<RenderSystem>(size_t order);
template <> PhysicsSystem *World::addSystem<PhysicsSystem>(size_t order);
template <> CleanUpSystem *World::addSystem<CleanUpSystem>(size_t order);
template <> DebugSystem *World::addSystem<DebugSystem>(size_t order);
template <> GizmoSystem *World::addSystem<GizmoSystem>(size_t order);
template <> FreeLookCameraSystem *World::addSystem<FreeLookCameraSystem>(size_t order);
template <> TerrainSystem *World::addSystem<TerrainSystem>(size_t order);
template <> RenderSystem *World::getSystemByIndex<RenderSystem>(size_t index) const;
template <> PhysicsSystem *World::getSystemByIndex<PhysicsSystem>(size_t index) const;
template <> CleanUpSystem *World::getSystemByIndex<CleanUpSystem>(size_t index) const;
template <> DebugSystem *World::getSystemByIndex<DebugSystem>(size_t index) const;
template <> GizmoSystem *World::getSystemByIndex<GizmoSystem>(size_t index) const;
template <> FreeLookCameraSystem *World::getSystemByIndex<FreeLookCameraSystem>(size_t index) const;
template <> TerrainSystem *World::getSystemByIndex<TerrainSystem>(size_t index) const;
template <> RenderSystem *World::getSystemById<RenderSystem>(const Guid &systemId) const;
template <> PhysicsSystem *World::getSystemById<PhysicsSystem>(const Guid &systemId) const;
template <> CleanUpSystem *World::getSystemById<CleanUpSystem>(const Guid &systemId) const;
template <> DebugSystem *World::getSystemById<DebugSystem>(const Guid &systemId) const;
template <> GizmoSystem *World::getSystemById<GizmoSystem>(const Guid &systemId) const;
template <> FreeLookCameraSystem *World::getSystemById<FreeLookCameraSystem>(const Guid &systemId) const;
template <> TerrainSystem *World::getSystemById<TerrainSystem>(const Guid &systemId) const;
template <> Mesh *World::getAssetByIndex<Mesh>(size_t index) const;
template <> Material *World::getAssetByIndex<Material>(size_t index) const;
template <> Shader *World::getAssetByIndex<Shader>(size_t index) const;
template <> Texture2D *World::getAssetByIndex<Texture2D>(size_t index) const;
template <> Texture3D *World::getAssetByIndex<Texture3D>(size_t index) const;
template <> Cubemap *World::getAssetByIndex<Cubemap>(size_t index) const;
template <> RenderTexture *World::getAssetByIndex<RenderTexture>(size_t index) const;
template <> Font *World::getAssetByIndex<Font>(size_t index) const;
template <> Sprite *World::getAssetByIndex<Sprite>(size_t index) const;
template <> Mesh *World::getAssetById<Mesh>(const Guid &assetId) const;
template <> Material *World::getAssetById<Material>(const Guid &assetId) const;
template <> Shader *World::getAssetById<Shader>(const Guid &assetId) const;
template <> Texture2D *World::getAssetById<Texture2D>(const Guid &assetId) const;
template <> Texture3D *World::getAssetById<Texture3D>(const Guid &assetId) const;
template <> Cubemap *World::getAssetById<Cubemap>(const Guid &assetId) const;
template <> RenderTexture *World::getAssetById<RenderTexture>(const Guid &assetId) const;
template <> Font *World::getAssetById<Font>(const Guid &assetId) const;
template <> Sprite *World::getAssetById<Sprite>(const Guid &assetId) const;
template <> Transform *World::getComponentByIndex<Transform>(size_t index) const;
template <> MeshRenderer *World::getComponentByIndex<MeshRenderer>(size_t index) const;
template <> SpriteRenderer *World::getComponentByIndex<SpriteRenderer>(size_t index) const;
template <> LineRenderer *World::getComponentByIndex<LineRenderer>(size_t index) const;
template <> Rigidbody *World::getComponentByIndex<Rigidbody>(size_t index) const;
template <> Camera *World::getComponentByIndex<Camera>(size_t index) const;
template <> Light *World::getComponentByIndex<Light>(size_t index) const;
template <> SphereCollider *World::getComponentByIndex<SphereCollider>(size_t index) const;
template <> BoxCollider *World::getComponentByIndex<BoxCollider>(size_t index) const;
template <> CapsuleCollider *World::getComponentByIndex<CapsuleCollider>(size_t index) const;
template <> MeshCollider *World::getComponentByIndex<MeshCollider>(size_t index) const;
template <> Terrain *World::getComponentByIndex<Terrain>(size_t index) const;
template <> Transform *World::getComponentById<Transform>(const Guid &componentId) const;
template <> MeshRenderer *World::getComponentById<MeshRenderer>(const Guid &componentId) const;
template <> SpriteRenderer *World::getComponentById<SpriteRenderer>(const Guid &componentId) const;
template <> LineRenderer *World::getComponentById<LineRenderer>(const Guid &componentId) const;
template <> Rigidbody *World::getComponentById<Rigidbody>(const Guid &componentId) const;
template <> Camera *World::getComponentById<Camera>(const Guid &componentId) const;
template <> Light *World::getComponentById<Light>(const Guid &componentId) const;
template <> SphereCollider *World::getComponentById<SphereCollider>(const Guid &componentId) const;
template <> BoxCollider *World::getComponentById<BoxCollider>(const Guid &componentId) const;
template <> CapsuleCollider *World::getComponentById<CapsuleCollider>(const Guid &componentId) const;
template <> MeshCollider *World::getComponentById<MeshCollider>(const Guid &componentId) const;
template <> Terrain *World::getComponentById<Terrain>(const Guid &componentId) const;
template <> Mesh *World::createAsset<Mesh>();
template <> Mesh *World::createAsset<Mesh>(const Guid &id);
template <> Material *World::createAsset<Material>();
template <> Material *World::createAsset<Material>(const Guid &id);
template <> Shader *World::createAsset<Shader>();
template <> Shader *World::createAsset<Shader>(const Guid &id);
template <> Texture2D *World::createAsset<Texture2D>();
template <> Texture2D *World::createAsset<Texture2D>(const Guid &id);
template <> Texture3D *World::createAsset<Texture3D>();
template <> Texture3D *World::createAsset<Texture3D>(const Guid &id);
template <> Cubemap *World::createAsset<Cubemap>();
template <> Cubemap *World::createAsset<Cubemap>(const Guid &id);
template <> RenderTexture *World::createAsset<RenderTexture>();
template <> RenderTexture *World::createAsset<RenderTexture>(const Guid &id);
template <> Font *World::createAsset<Font>();
template <> Font *World::createAsset<Font>(const Guid &id);
template <> Sprite *World::createAsset<Sprite>();
template <> Sprite *World::createAsset<Sprite>(const Guid &id);
template <>
Transform *World::getComponentById_impl<Transform>(const PoolAllocator<Transform> *allocator,
                                                   const Guid &componentId) const;
template <>
MeshRenderer *World::getComponentById_impl<MeshRenderer>(const PoolAllocator<MeshRenderer> *allocator,
                                                         const Guid &componentId) const;
template <>
SpriteRenderer *World::getComponentById_impl<SpriteRenderer>(const PoolAllocator<SpriteRenderer> *allocator,
                                                             const Guid &componentId) const;
template <>
LineRenderer *World::getComponentById_impl<LineRenderer>(const PoolAllocator<LineRenderer> *allocator,
                                                         const Guid &componentId) const;
template <>
Rigidbody *World::getComponentById_impl<Rigidbody>(const PoolAllocator<Rigidbody> *allocator,
                                                   const Guid &componentId) const;
template <>
Camera *World::getComponentById_impl<Camera>(const PoolAllocator<Camera> *allocator, const Guid &componentId) const;
template <>
Light *World::getComponentById_impl<Light>(const PoolAllocator<Light> *allocator, const Guid &componentId) const;
template <>
SphereCollider *World::getComponentById_impl<SphereCollider>(const PoolAllocator<SphereCollider> *allocator,
                                                             const Guid &componentId) const;
template <>
BoxCollider *World::getComponentById_impl<BoxCollider>(const PoolAllocator<BoxCollider> *allocator,
                                                       const Guid &componentId) const;
template <>
CapsuleCollider *World::getComponentById_impl<CapsuleCollider>(const PoolAllocator<CapsuleCollider> *allocator,
                                                               const Guid &componentId) const;
template <>
MeshCollider *World::getComponentById_impl<MeshCollider>(const PoolAllocator<MeshCollider> *allocator,
                                                         const Guid &componentId) const;
template <>
Terrain *World::getComponentById_impl<Terrain>(const PoolAllocator<Terrain> *allocator,
                                                         const Guid &componentId) const;
template <> Mesh *World::getAssetById_impl<Mesh>(const PoolAllocator<Mesh> *allocator, const Guid &assetId) const;
template <>
Material *World::getAssetById_impl<Material>(const PoolAllocator<Material> *allocator, const Guid &assetId) const;
template <> Shader *World::getAssetById_impl<Shader>(const PoolAllocator<Shader> *allocator, const Guid &assetId) const;
template <>
Texture2D *World::getAssetById_impl<Texture2D>(const PoolAllocator<Texture2D> *allocator, const Guid &assetId) const;
template <>
Texture3D *World::getAssetById_impl<Texture3D>(const PoolAllocator<Texture3D> *allocator, const Guid &assetId) const;
template <>
Cubemap *World::getAssetById_impl<Cubemap>(const PoolAllocator<Cubemap> *allocator, const Guid &assetId) const;
template <>
RenderTexture *World::getAssetById_impl<RenderTexture>(const PoolAllocator<RenderTexture> *allocator,
                                                       const Guid &assetId) const;
template <> Font *World::getAssetById_impl<Font>(const PoolAllocator<Font> *allocator, const Guid &assetId) const;
template <> Sprite *World::getAssetById_impl<Sprite>(const PoolAllocator<Sprite> *allocator, const Guid &assetId) const;
template <>
RenderSystem *World::getSystemById_impl<RenderSystem>(const PoolAllocator<RenderSystem> *allocator,
                                                      const Guid &assetId) const;
template <>
PhysicsSystem *World::getSystemById_impl<PhysicsSystem>(const PoolAllocator<PhysicsSystem> *allocator,
                                                        const Guid &assetId) const;
template <>
CleanUpSystem *World::getSystemById_impl<CleanUpSystem>(const PoolAllocator<CleanUpSystem> *allocator,
                                                        const Guid &assetId) const;
template <>
DebugSystem *World::getSystemById_impl<DebugSystem>(const PoolAllocator<DebugSystem> *allocator,
                                                    const Guid &assetId) const;
template <>
GizmoSystem *World::getSystemById_impl<GizmoSystem>(const PoolAllocator<GizmoSystem> *allocator,
                                                    const Guid &assetId) const;
template <>
FreeLookCameraSystem *World::getSystemById_impl<FreeLookCameraSystem>(
    const PoolAllocator<FreeLookCameraSystem> *allocator, const Guid &assetId) const;
template <>
TerrainSystem *World::getSystemById_impl<TerrainSystem>(
    const PoolAllocator<TerrainSystem> *allocator, const Guid &assetId) const;
template <> void World::addIdToGlobalIndexMap_impl<Scene>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Entity>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Transform>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<MeshRenderer>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<SpriteRenderer>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<LineRenderer>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Rigidbody>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Camera>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Light>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<SphereCollider>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<BoxCollider>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<CapsuleCollider>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<MeshCollider>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Terrain>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Mesh>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Material>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Shader>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Texture2D>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Texture3D>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Cubemap>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<RenderTexture>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Font>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Sprite>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<RenderSystem>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<PhysicsSystem>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<CleanUpSystem>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<DebugSystem>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<GizmoSystem>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<FreeLookCameraSystem>(const Guid &id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<TerrainSystem>(const Guid &id, int index, int type);

} // namespace PhysicsEngine

#endif