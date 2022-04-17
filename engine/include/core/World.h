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

    std::unordered_map<Id, std::unordered_map<Id, std::vector<ShaderUniform>>> mMaterialUniformCache;

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
    bool writeAssetToYAML(const std::string &filePath, Id assetId) const;
    bool writeSceneToYAML(const std::string &filePath, Id sceneId) const;

    void latentDestroyEntitiesInWorld();
    void immediateDestroyEntitiesInWorld();

    std::vector<ShaderUniform> getCachedMaterialUniforms(Id materialId, Id shaderId);
    void cacheMaterialUniforms(Id materialId, Id shaderId, const std::vector<ShaderUniform>& uniforms);

    size_t getNumberOfScenes() const;
    size_t getNumberOfEntities() const;
    size_t getNumberOfNonHiddenEntities() const;
    size_t getNumberOfUpdatingSystems() const;
    Mesh *getPrimtiveMesh(PrimitiveType type) const;
    Entity *createPrimitive(PrimitiveType type);
    Entity *createNonPrimitive(Id meshId);
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

    template <typename T> T *getComponent(Id entityId) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return getComponent_impl<T>(getComponentAllocator_impl<T>(), entityId);
    }

    template <typename T> T *addComponent(Id entityId)
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
    Scene *getSceneById(Id sceneId) const;
    Entity *getEntityByIndex(size_t index) const;
    Entity *getEntityById(Id entityId) const;
    System *getSystemByUpdateOrder(size_t order) const;

    template <typename T> T *getSystemByIndex(size_t index) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return getSystemByIndex_impl(getSystemAllocator_impl<T>(), index);
    }

    template <typename T> T *getSystemById(Id systemId) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return getSystemById_impl<T>(getSystemAllocator_impl<T>(), systemId);
    }

    template <typename T> T *getAssetByIndex(size_t index) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return getAssetByIndex_impl(getAssetAllocator_impl<T>(), index);
    }

    template <typename T> T *getAssetById(Id assetId) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return getAssetById_impl<T>(getAssetAllocator_impl<T>(), assetId);
    }

    template <typename T> T *getComponentByIndex(size_t index) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return getComponentByIndex_impl(getComponentAllocator_impl<T>(), index);
    }

    template <typename T> T *getComponentById(Id componentId) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return getComponentById_impl<T>(getComponentAllocator_impl<T>(), componentId);
    }

    template <typename T> T *createAsset()
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return createAsset_impl<T>(getAssetOrAddAllocator_impl<T>());
    }

    Guid getGuidOf(Id id) const;
    Id getIdOf(const Guid &guid) const;
    int getIndexOf(Id id) const;
    int getTypeOf(Id id) const;

    Scene *createScene();
    Entity *createEntity();
    Entity *createEntity(const std::string& name);

    void latentDestroyEntity(Id entityId);
    void immediateDestroyEntity(Id entityId);
    void latentDestroyComponent(Id entityId, Id componentId, int componentType);
    void immediateDestroyComponent(Id entityId, Id componentId, int componentType);
    void latentDestroyAsset(Id assetId, int assetType);
    void immediateDestroyAsset(Id assetId, int assetType);

    bool isMarkedForLatentDestroy(Id id);
    void clearIdsMarkedCreatedOrDestroyed();

    std::vector<std::pair<Id, int>> getComponentsOnEntity(Id entityId) const;

    std::vector<Id> getEntityIdsMarkedCreated() const;
    std::vector<Id> getEntityIdsMarkedLatentDestroy() const;
    std::vector<std::tuple<Id, Id, int>> getComponentIdsMarkedCreated() const;
    std::vector<std::tuple<Id, Id, int>> getComponentIdsMarkedLatentDestroy() const;

    /*std::string getAssetFilepath(const Guid &assetId) const;
    std::string getSceneFilepath(const Guid &sceneId) const;

    Guid getAssetId(const std::string& filepath) const;
    Guid getSceneId(const std::string& filepath) const;*/

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

    template <typename T> T *getComponent_impl(const PoolAllocator<T> *allocator, Id entityId) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        assert(allocator != nullptr);

        std::vector<std::pair<Id, int>> componentsOnEntity = getComponentsOnEntity(entityId);

        for (size_t i = 0; i < componentsOnEntity.size(); i++)
        {
            if (ComponentType<T>::type == componentsOnEntity[i].second)
            {
                std::unordered_map<Id, int>::const_iterator it =
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

    template <typename T> T *addComponent_impl(PoolAllocator<T> *allocator, Id entityId)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        assert(allocator != nullptr);

        if (getTypeOf(entityId) != EntityType<Entity>::type)
        {
            return nullptr;
        }

        int componentGlobalIndex = (int)allocator->getCount();
        int componentType = ComponentType<T>::type;

        T *component = allocator->construct(this);

        if (component != nullptr)
        {
            component->mEntityId = entityId;

            addIdToGlobalIndexMap_impl<T>(component->getId(), componentGlobalIndex, componentType);

            mIdState.mEntityIdToComponentIds[entityId].push_back(std::make_pair(component->getId(), componentType));

            mIdState.mComponentIdsMarkedCreated.push_back(std::make_tuple(entityId, component->getId(), componentType));
        }

        return component;
    }

    template <typename T> T *addSystem_impl(PoolAllocator<T> *allocator, size_t order)
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        int systemGlobalIndex = (int)allocator->getCount();
        int systemType = SystemType<T>::type;
        
        T *system = allocator->construct(this);

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

    template <typename T> T *getSystemById_impl(const PoolAllocator<T> *allocator, Id systemId) const
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

    template <typename T> T *getAssetById_impl(const PoolAllocator<T> *allocator, Id assetId) const
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

    template <typename T> T *getComponentById_impl(const PoolAllocator<T> *allocator, Id componentId) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        if (allocator == nullptr || ComponentType<T>::type != getTypeOf(componentId))
        {
            return nullptr;
        }

        return getById_impl<T>(mIdState.mIdToGlobalIndex, allocator, componentId);
    }

    template <typename T> T *createAsset_impl(PoolAllocator<T> *allocator)
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        int index = (int)allocator->getCount();
        int type = AssetType<T>::type;

        T *asset = allocator->construct(this);

        if (asset != nullptr)
        {
            addIdToGlobalIndexMap_impl<T>(asset->getId(), index, type);
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
    T *getById_impl(const std::unordered_map<Id, int> &idToIndexMap, const PoolAllocator<T> *allocator, Id id) const
    {
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

    template <typename T> void addIdToGlobalIndexMap_impl(Id id, int index, int type)
    {
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <int N> void removeIdToGlobalIndexMap_impl(Id id, int type)
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
template <> Transform *World::getComponent<Transform>(Id entityId) const;
template <> MeshRenderer *World::getComponent<MeshRenderer>(Id entityId) const;
template <> SpriteRenderer *World::getComponent<SpriteRenderer>(Id entityId) const;
template <> LineRenderer *World::getComponent<LineRenderer>(Id entityId) const;
template <> Rigidbody *World::getComponent<Rigidbody>(Id entityId) const;
template <> Camera *World::getComponent<Camera>(Id entityId) const;
template <> Light *World::getComponent<Light>(Id entityId) const;
template <> SphereCollider *World::getComponent<SphereCollider>(Id entityId) const;
template <> BoxCollider *World::getComponent<BoxCollider>(Id entityId) const;
template <> CapsuleCollider *World::getComponent<CapsuleCollider>(Id entityId) const;
template <> MeshCollider *World::getComponent<MeshCollider>(Id entityId) const;
template <> Terrain *World::getComponent<Terrain>(Id entityId) const;
template <> MeshRenderer *World::addComponent<MeshRenderer>(Id entityId);
template <> SpriteRenderer *World::addComponent<SpriteRenderer>(Id entityId);
template <> LineRenderer *World::addComponent<LineRenderer>(Id entityId);
template <> Rigidbody *World::addComponent<Rigidbody>(Id entityId);
template <> Camera *World::addComponent<Camera>(Id entityId);
template <> Light *World::addComponent<Light>(Id entityId);
template <> SphereCollider *World::addComponent<SphereCollider>(Id entityId);
template <> BoxCollider *World::addComponent<BoxCollider>(Id entityId);
template <> CapsuleCollider *World::addComponent<CapsuleCollider>(Id entityId);
template <> MeshCollider *World::addComponent<MeshCollider>(Id entityId);
template <> Terrain *World::addComponent<Terrain>(Id entityId);
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
template <> RenderSystem *World::getSystemById<RenderSystem>(Id systemId) const;
template <> PhysicsSystem *World::getSystemById<PhysicsSystem>(Id systemId) const;
template <> CleanUpSystem *World::getSystemById<CleanUpSystem>(Id systemId) const;
template <> DebugSystem *World::getSystemById<DebugSystem>(Id systemId) const;
template <> GizmoSystem *World::getSystemById<GizmoSystem>(Id systemId) const;
template <> FreeLookCameraSystem *World::getSystemById<FreeLookCameraSystem>(Id systemId) const;
template <> TerrainSystem *World::getSystemById<TerrainSystem>(Id systemId) const;
template <> Mesh *World::getAssetByIndex<Mesh>(size_t index) const;
template <> Material *World::getAssetByIndex<Material>(size_t index) const;
template <> Shader *World::getAssetByIndex<Shader>(size_t index) const;
template <> Texture2D *World::getAssetByIndex<Texture2D>(size_t index) const;
template <> Texture3D *World::getAssetByIndex<Texture3D>(size_t index) const;
template <> Cubemap *World::getAssetByIndex<Cubemap>(size_t index) const;
template <> RenderTexture *World::getAssetByIndex<RenderTexture>(size_t index) const;
template <> Font *World::getAssetByIndex<Font>(size_t index) const;
template <> Sprite *World::getAssetByIndex<Sprite>(size_t index) const;
template <> Mesh *World::getAssetById<Mesh>(Id assetId) const;
template <> Material *World::getAssetById<Material>(Id assetId) const;
template <> Shader *World::getAssetById<Shader>(Id assetId) const;
template <> Texture2D *World::getAssetById<Texture2D>(Id assetId) const;
template <> Texture3D *World::getAssetById<Texture3D>(Id assetId) const;
template <> Cubemap *World::getAssetById<Cubemap>(Id assetId) const;
template <> RenderTexture *World::getAssetById<RenderTexture>(Id assetId) const;
template <> Font *World::getAssetById<Font>(Id assetId) const;
template <> Sprite *World::getAssetById<Sprite>(Id assetId) const;
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
template <> Transform *World::getComponentById<Transform>(Id componentId) const;
template <> MeshRenderer *World::getComponentById<MeshRenderer>(Id componentId) const;
template <> SpriteRenderer *World::getComponentById<SpriteRenderer>(Id componentId) const;
template <> LineRenderer *World::getComponentById<LineRenderer>(Id componentId) const;
template <> Rigidbody *World::getComponentById<Rigidbody>(Id componentId) const;
template <> Camera *World::getComponentById<Camera>(Id componentId) const;
template <> Light *World::getComponentById<Light>(Id componentId) const;
template <> SphereCollider *World::getComponentById<SphereCollider>(Id componentId) const;
template <> BoxCollider *World::getComponentById<BoxCollider>(Id componentId) const;
template <> CapsuleCollider *World::getComponentById<CapsuleCollider>(Id componentId) const;
template <> MeshCollider *World::getComponentById<MeshCollider>(Id componentId) const;
template <> Terrain *World::getComponentById<Terrain>(Id componentId) const;
template <> Mesh *World::createAsset<Mesh>();
template <> Material *World::createAsset<Material>();
template <> Shader *World::createAsset<Shader>();
template <> Texture2D *World::createAsset<Texture2D>();
template <> Texture3D *World::createAsset<Texture3D>();
template <> Cubemap *World::createAsset<Cubemap>();
template <> RenderTexture *World::createAsset<RenderTexture>();
template <> Font *World::createAsset<Font>();
template <> Sprite *World::createAsset<Sprite>();
template <>
Transform *World::getComponentById_impl<Transform>(const PoolAllocator<Transform> *allocator,
                                                   Id componentId) const;
template <>
MeshRenderer *World::getComponentById_impl<MeshRenderer>(const PoolAllocator<MeshRenderer> *allocator,
                                                         Id componentId) const;
template <>
SpriteRenderer *World::getComponentById_impl<SpriteRenderer>(const PoolAllocator<SpriteRenderer> *allocator,
                                                             Id componentId) const;
template <>
LineRenderer *World::getComponentById_impl<LineRenderer>(const PoolAllocator<LineRenderer> *allocator,
                                                         Id componentId) const;
template <>
Rigidbody *World::getComponentById_impl<Rigidbody>(const PoolAllocator<Rigidbody> *allocator,
                                                   Id componentId) const;
template <>
Camera *World::getComponentById_impl<Camera>(const PoolAllocator<Camera> *allocator, Id componentId) const;
template <>
Light *World::getComponentById_impl<Light>(const PoolAllocator<Light> *allocator, Id componentId) const;
template <>
SphereCollider *World::getComponentById_impl<SphereCollider>(const PoolAllocator<SphereCollider> *allocator,
                                                             Id componentId) const;
template <>
BoxCollider *World::getComponentById_impl<BoxCollider>(const PoolAllocator<BoxCollider> *allocator,
                                                       Id componentId) const;
template <>
CapsuleCollider *World::getComponentById_impl<CapsuleCollider>(const PoolAllocator<CapsuleCollider> *allocator,
                                                               Id componentId) const;
template <>
MeshCollider *World::getComponentById_impl<MeshCollider>(const PoolAllocator<MeshCollider> *allocator,
                                                         Id componentId) const;
template <>
Terrain *World::getComponentById_impl<Terrain>(const PoolAllocator<Terrain> *allocator, Id componentId) const;
template <> Mesh *World::getAssetById_impl<Mesh>(const PoolAllocator<Mesh> *allocator, Id assetId) const;
template <>
Material *World::getAssetById_impl<Material>(const PoolAllocator<Material> *allocator, Id assetId) const;
template <> Shader *World::getAssetById_impl<Shader>(const PoolAllocator<Shader> *allocator, Id assetId) const;
template <>
Texture2D *World::getAssetById_impl<Texture2D>(const PoolAllocator<Texture2D> *allocator, Id assetId) const;
template <>
Texture3D *World::getAssetById_impl<Texture3D>(const PoolAllocator<Texture3D> *allocator, Id assetId) const;
template <>
Cubemap *World::getAssetById_impl<Cubemap>(const PoolAllocator<Cubemap> *allocator, Id assetId) const;
template <>
RenderTexture *World::getAssetById_impl<RenderTexture>(const PoolAllocator<RenderTexture> *allocator,
                                                       Id assetId) const;
template <> Font *World::getAssetById_impl<Font>(const PoolAllocator<Font> *allocator, Id assetId) const;
template <> Sprite *World::getAssetById_impl<Sprite>(const PoolAllocator<Sprite> *allocator, Id assetId) const;
template <>
RenderSystem *World::getSystemById_impl<RenderSystem>(const PoolAllocator<RenderSystem> *allocator,
                                                      Id assetId) const;
template <>
PhysicsSystem *World::getSystemById_impl<PhysicsSystem>(const PoolAllocator<PhysicsSystem> *allocator,
                                                        Id assetId) const;
template <>
CleanUpSystem *World::getSystemById_impl<CleanUpSystem>(const PoolAllocator<CleanUpSystem> *allocator,
                                                        Id assetId) const;
template <>
DebugSystem *World::getSystemById_impl<DebugSystem>(const PoolAllocator<DebugSystem> *allocator,
                                                    Id assetId) const;
template <>
GizmoSystem *World::getSystemById_impl<GizmoSystem>(const PoolAllocator<GizmoSystem> *allocator,
                                                    Id assetId) const;
template <>
FreeLookCameraSystem *World::getSystemById_impl<FreeLookCameraSystem>(
    const PoolAllocator<FreeLookCameraSystem> *allocator, Id assetId) const;
template <>
TerrainSystem *World::getSystemById_impl<TerrainSystem>(
    const PoolAllocator<TerrainSystem> *allocator, Id assetId) const;
template <> void World::addIdToGlobalIndexMap_impl<Scene>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Entity>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Transform>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<MeshRenderer>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<SpriteRenderer>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<LineRenderer>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Rigidbody>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Camera>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Light>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<SphereCollider>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<BoxCollider>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<CapsuleCollider>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<MeshCollider>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Terrain>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Mesh>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Material>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Shader>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Texture2D>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Texture3D>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Cubemap>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<RenderTexture>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Font>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<Sprite>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<RenderSystem>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<PhysicsSystem>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<CleanUpSystem>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<DebugSystem>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<GizmoSystem>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<FreeLookCameraSystem>(Id id, int index, int type);
template <> void World::addIdToGlobalIndexMap_impl<TerrainSystem>(Id id, int index, int type);

} // namespace PhysicsEngine

#endif