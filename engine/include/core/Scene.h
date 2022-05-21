#ifndef SCENE_H__
#define SCENE_H__

#include <string>
#include <unordered_map>

#include "Object.h"

#include "Guid.h"
#include "PoolAllocator.h"
#include "Entity.h"
#include "WorldPrimitives.h"

#include "../components/BoxCollider.h"
#include "../components/Camera.h"
#include "../components/CapsuleCollider.h"
#include "../components/Light.h"
#include "../components/LineRenderer.h"
#include "../components/MeshCollider.h"
#include "../components/MeshRenderer.h"
#include "../components/Rigidbody.h"
#include "../components/SphereCollider.h"
#include "../components/SpriteRenderer.h"
#include "../components/Transform.h"
#include "../components/Terrain.h"

namespace PhysicsEngine
{
// Simple structs used for grouping scene id to global index maps when passing to functions
struct SceneIdState
{
    // internal world entity id state
    std::unordered_map<Guid, int> mEntityIdToGlobalIndex;

    // internal world components id state
    std::unordered_map<Guid, int> mTransformIdToGlobalIndex;
    std::unordered_map<Guid, int> mMeshRendererIdToGlobalIndex;
    std::unordered_map<Guid, int> mSpriteRendererIdToGlobalIndex;
    std::unordered_map<Guid, int> mLineRendererIdToGlobalIndex;
    std::unordered_map<Guid, int> mRigidbodyIdToGlobalIndex;
    std::unordered_map<Guid, int> mCameraIdToGlobalIndex;
    std::unordered_map<Guid, int> mLightIdToGlobalIndex;
    std::unordered_map<Guid, int> mSphereColliderIdToGlobalIndex;
    std::unordered_map<Guid, int> mBoxColliderIdToGlobalIndex;
    std::unordered_map<Guid, int> mCapsuleColliderIdToGlobalIndex;
    std::unordered_map<Guid, int> mMeshColliderIdToGlobalIndex;
    std::unordered_map<Guid, int> mTerrainIdToGlobalIndex;

    // world id state for all entity and components
    std::unordered_map<Guid, int> mIdToGlobalIndex;
    std::unordered_map<Guid, int> mIdToType;

    // entity ids to component ids
    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>> mEntityIdToComponentIds;

    // entity creation/deletion state
    std::vector<Guid> mEntityIdsMarkedCreated;
    std::vector<Guid> mEntityIdsMarkedLatentDestroy;
    std::vector<std::pair<Guid, int>> mEntityIdsMarkedMoved;

    // component create/deletion state
    std::vector<std::tuple<Guid, Guid, int>> mComponentIdsMarkedCreated;
    std::vector<std::tuple<Guid, Guid, int>> mComponentIdsMarkedLatentDestroy;
    std::vector<std::tuple<Guid, int, int>> mComponentIdsMarkedMoved;
};

// Simple structs used for grouping scene allocators when passing to functions
struct SceneAllocators
{
    // internal entity allocator
    PoolAllocator<Entity> mEntityAllocator;

    // internal component allocators
    PoolAllocator<Transform> mTransformAllocator;
    PoolAllocator<MeshRenderer> mMeshRendererAllocator;
    PoolAllocator<SpriteRenderer> mSpriteRendererAllocator;
    PoolAllocator<LineRenderer> mLineRendererAllocator;
    PoolAllocator<Rigidbody> mRigidbodyAllocator;
    PoolAllocator<Camera> mCameraAllocator;
    PoolAllocator<Light> mLightAllocator;
    PoolAllocator<SphereCollider> mSphereColliderAllocator;
    PoolAllocator<BoxCollider> mBoxColliderAllocator;
    PoolAllocator<CapsuleCollider> mCapsuleColliderAllocator;
    PoolAllocator<MeshCollider> mMeshColliderAllocator;
    PoolAllocator<Terrain> mTerrainAllocator;
};

class Scene : public Object
{
  private:
    // allocators for entities and components
    SceneAllocators mAllocators;

    // id state for entities and components
    SceneIdState mIdState;

    std::string mName;
    std::string mVersion;

  public:
    Scene(World *world);
    Scene(World *world, const Guid& id);
    ~Scene();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    bool writeToYAML(const std::string &filepath) const;

    void load(const std::string &filepath);

    std::string getName() const;

    size_t getNumberOfEntities() const;
    size_t getNumberOfNonHiddenEntities() const;

    int getIndexOf(const Guid &id) const;
    int getTypeOf(const Guid &id) const;
    Entity *getEntityByIndex(size_t index) const;
    Entity *getEntityById(const Guid &entityId) const;
    Component *getComponentById(const Guid &componentId, int type) const;
    //Component *addComponent(const Guid &entityId, int type);
    Component *addComponent(const YAML::Node &in, int type);

    Entity *createEntity();
    Entity *createEntity(const YAML::Node &in);
    Entity *createEntity(const std::string &name);
    Entity *createPrimitive(PrimitiveType type);
    Entity *createNonPrimitive(const Guid &meshId);
    Entity *createLight(LightType type);
    Entity *createCamera();

    void latentDestroyEntitiesInScene();
    void immediateDestroyEntitiesInScene();

    void latentDestroyEntity(const Guid &entityId);
    void immediateDestroyEntity(const Guid &entityId);
    void latentDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType);
    void immediateDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType);

    bool isMarkedForLatentDestroy(const Guid &id);
    void clearIdsMarkedCreatedOrDestroyed();

    std::vector<std::pair<Guid, int>> getComponentsOnEntity(const Guid &entityId) const;

    std::vector<Guid> getEntityIdsMarkedCreated() const;
    std::vector<Guid> getEntityIdsMarkedLatentDestroy() const;
    std::vector<std::tuple<Guid, Guid, int>> getComponentIdsMarkedCreated() const;
    std::vector<std::tuple<Guid, Guid, int>> getComponentIdsMarkedLatentDestroy() const;

    template <typename T> size_t getNumberOfComponents() const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");
        return 0;
    }

    template <> size_t getNumberOfComponents<Transform>() const
    {
        return mAllocators.mTransformAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<MeshRenderer>() const
    {
        return mAllocators.mMeshRendererAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<SpriteRenderer>() const
    {
        return mAllocators.mSpriteRendererAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<LineRenderer>() const
    {
        return mAllocators.mLineRendererAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<Rigidbody>() const
    {
        return mAllocators.mRigidbodyAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<Camera>() const
    {
        return mAllocators.mCameraAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<Light>() const
    {
        return mAllocators.mLightAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<SphereCollider>() const
    {
        return mAllocators.mSphereColliderAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<BoxCollider>() const
    {
        return mAllocators.mBoxColliderAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<CapsuleCollider>() const
    {
        return mAllocators.mCapsuleColliderAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<MeshCollider>() const
    {
        return mAllocators.mMeshColliderAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<Terrain>() const
    {
        return mAllocators.mTerrainAllocator.getCount();
    }

    template <typename T> T *getComponentByIndex(size_t index) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");
        return nullptr;
    }

    template <> Transform *getComponentByIndex<Transform>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mTransformAllocator, index);
    }

    template <> MeshRenderer *getComponentByIndex<MeshRenderer>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mMeshRendererAllocator, index);
    }

    template <> SpriteRenderer *getComponentByIndex<SpriteRenderer>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mSpriteRendererAllocator, index);
    }

    template <> LineRenderer *getComponentByIndex<LineRenderer>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mLineRendererAllocator, index);
    }

    template <> Rigidbody *getComponentByIndex<Rigidbody>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mRigidbodyAllocator, index);
    }

    template <> Camera *getComponentByIndex<Camera>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mCameraAllocator, index);
    }

    template <> Light *getComponentByIndex<Light>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mLightAllocator, index);
    }

    template <> SphereCollider *getComponentByIndex<SphereCollider>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mSphereColliderAllocator, index);
    }

    template <> BoxCollider *getComponentByIndex<BoxCollider>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mBoxColliderAllocator, index);
    }

    template <> CapsuleCollider *getComponentByIndex<CapsuleCollider>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mCapsuleColliderAllocator, index);
    }

    template <> MeshCollider *getComponentByIndex<MeshCollider>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mMeshColliderAllocator, index);
    }

    template <> Terrain *getComponentByIndex<Terrain>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mTerrainAllocator, index);
    }

    template <typename T> T *getComponent(const Guid &entityId) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");
        return nullptr;
    }

    template <> Transform *getComponent<Transform>(const Guid &entityId) const
    {
        // Transform occurs at same index as its entity since all entities have a transform
        return getComponentByIndex<Transform>(getIndexOf(entityId));
    }

    template <> MeshRenderer *getComponent<MeshRenderer>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mMeshRendererAllocator, entityId);
    }

    template <> SpriteRenderer *getComponent<SpriteRenderer>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mSpriteRendererAllocator, entityId);
    }

    template <> LineRenderer *getComponent<LineRenderer>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mLineRendererAllocator, entityId);
    }

    template <> Rigidbody *getComponent<Rigidbody>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mRigidbodyAllocator, entityId);
    }

    template <> Camera *getComponent<Camera>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mCameraAllocator, entityId);
    }

    template <> Light *getComponent<Light>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mLightAllocator, entityId);
    }

    template <> SphereCollider *getComponent<SphereCollider>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mSphereColliderAllocator, entityId);
    }

    template <> BoxCollider *getComponent<BoxCollider>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mBoxColliderAllocator, entityId);
    }

    template <> CapsuleCollider *getComponent<CapsuleCollider>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mCapsuleColliderAllocator, entityId);
    }

    template <> MeshCollider *getComponent<MeshCollider>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mMeshColliderAllocator, entityId);
    }

    template <> Terrain *getComponent<Terrain>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mTerrainAllocator, entityId);
    }

    template <typename T> T *addComponent(const Guid &entityId)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");
        return nullptr;
    }

    template <> MeshRenderer *addComponent<MeshRenderer>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mMeshRendererAllocator, entityId);
    }

    template <> SpriteRenderer *addComponent<SpriteRenderer>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mSpriteRendererAllocator, entityId);
    }

    template <> LineRenderer *addComponent<LineRenderer>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mLineRendererAllocator, entityId);
    }

    template <> Rigidbody *addComponent<Rigidbody>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mRigidbodyAllocator, entityId);
    }

    template <> Camera *addComponent<Camera>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mCameraAllocator, entityId);
    }

    template <> Light *addComponent<Light>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mLightAllocator, entityId);
    }

    template <> SphereCollider *addComponent<SphereCollider>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mSphereColliderAllocator, entityId);
    }

    template <> BoxCollider *addComponent<BoxCollider>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mBoxColliderAllocator, entityId);
    }

    template <> CapsuleCollider *addComponent<CapsuleCollider>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mCapsuleColliderAllocator, entityId);
    }

    template <> MeshCollider *addComponent<MeshCollider>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mMeshColliderAllocator, entityId);
    }

    template <> Terrain *addComponent<Terrain>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mTerrainAllocator, entityId);
    }

    template <typename T> T *addComponent(const YAML::Node &in)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");
        return nullptr;
    }

    template <> Transform *addComponent<Transform>(const YAML::Node &in)
    {
        return addComponent_impl(&mAllocators.mTransformAllocator, in);
    }

    template <> MeshRenderer *addComponent<MeshRenderer>(const YAML::Node &in)
    {
        return addComponent_impl(&mAllocators.mMeshRendererAllocator, in);
    }

    template <> SpriteRenderer *addComponent<SpriteRenderer>(const YAML::Node &in)
    {
        return addComponent_impl(&mAllocators.mSpriteRendererAllocator, in);
    }

    template <> LineRenderer *addComponent<LineRenderer>(const YAML::Node &in)
    {
        return addComponent_impl(&mAllocators.mLineRendererAllocator, in);
    }

    template <> Rigidbody *addComponent<Rigidbody>(const YAML::Node &in)
    {
        return addComponent_impl(&mAllocators.mRigidbodyAllocator, in);
    }

    template <> Camera *addComponent<Camera>(const YAML::Node &in)
    {
        return addComponent_impl(&mAllocators.mCameraAllocator, in);
    }

    template <> Light *addComponent<Light>(const YAML::Node &in)
    {
        return addComponent_impl(&mAllocators.mLightAllocator, in);
    }

    template <> SphereCollider *addComponent<SphereCollider>(const YAML::Node &in)
    {
        return addComponent_impl(&mAllocators.mSphereColliderAllocator, in);
    }

    template <> BoxCollider *addComponent<BoxCollider>(const YAML::Node &in)
    {
        return addComponent_impl(&mAllocators.mBoxColliderAllocator, in);
    }

    template <> CapsuleCollider *addComponent<CapsuleCollider>(const YAML::Node &in)
    {
        return addComponent_impl(&mAllocators.mCapsuleColliderAllocator, in);
    }

    template <> MeshCollider *addComponent<MeshCollider>(const YAML::Node &in)
    {
        return addComponent_impl(&mAllocators.mMeshColliderAllocator, in);
    }

    template <> Terrain *addComponent<Terrain>(const YAML::Node &in)
    {
        return addComponent_impl(&mAllocators.mTerrainAllocator, in);
    }
   
    template <typename T> T *getComponentById(const Guid &componentId) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");
        return nullptr;
    }

    template <> Transform *getComponentById<Transform>(const Guid &componentId) const
    {
        return getComponentById_impl(mIdState.mTransformIdToGlobalIndex, &mAllocators.mTransformAllocator, componentId);
    }

    template <> MeshRenderer *getComponentById<MeshRenderer>(const Guid &componentId) const
    {
        return getComponentById_impl(mIdState.mMeshRendererIdToGlobalIndex, &mAllocators.mMeshRendererAllocator,
                                     componentId);
    }

    template <> SpriteRenderer *getComponentById<SpriteRenderer>(const Guid &componentId) const
    {
        return getComponentById_impl(mIdState.mSpriteRendererIdToGlobalIndex, &mAllocators.mSpriteRendererAllocator,
                                     componentId);
    }

    template <> LineRenderer *getComponentById<LineRenderer>(const Guid &componentId) const
    {
        return getComponentById_impl(mIdState.mLineRendererIdToGlobalIndex, &mAllocators.mLineRendererAllocator,
                                     componentId);
    }

    template <> Rigidbody *getComponentById<Rigidbody>(const Guid &componentId) const
    {
        return getComponentById_impl(mIdState.mRigidbodyIdToGlobalIndex, &mAllocators.mRigidbodyAllocator, componentId);
    }

    template <> Camera *getComponentById<Camera>(const Guid &componentId) const
    {
        return getComponentById_impl(mIdState.mCameraIdToGlobalIndex, &mAllocators.mCameraAllocator, componentId);
    }

    template <> Light *getComponentById<Light>(const Guid &componentId) const
    {
        return getComponentById_impl(mIdState.mLightIdToGlobalIndex, &mAllocators.mLightAllocator, componentId);
    }

    template <> SphereCollider *getComponentById<SphereCollider>(const Guid &componentId) const
    {
        return getComponentById_impl(mIdState.mSphereColliderIdToGlobalIndex, &mAllocators.mSphereColliderAllocator,
                                     componentId);
    }

    template <> BoxCollider *getComponentById<BoxCollider>(const Guid &componentId) const
    {
        return getComponentById_impl(mIdState.mBoxColliderIdToGlobalIndex, &mAllocators.mBoxColliderAllocator,
                                     componentId);
    }

    template <> CapsuleCollider *getComponentById<CapsuleCollider>(const Guid &componentId) const
    {
        return getComponentById_impl(mIdState.mCapsuleColliderIdToGlobalIndex, &mAllocators.mCapsuleColliderAllocator,
                                     componentId);
    }

    template <> MeshCollider *getComponentById<MeshCollider>(const Guid &componentId) const
    {
        return getComponentById_impl(mIdState.mMeshColliderIdToGlobalIndex, &mAllocators.mMeshColliderAllocator,
                                     componentId);
    }

    template <> Terrain *getComponentById<Terrain>(const Guid &componentId) const
    {
        return getComponentById_impl(mIdState.mTerrainIdToGlobalIndex, &mAllocators.mTerrainAllocator, componentId);
    }

  private:
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

    template <typename T> T *getComponentByIndex_impl(const PoolAllocator<T> *allocator, size_t index) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return allocator != nullptr ? allocator->get(index) : nullptr;
    }

    template <typename T>
    T *getComponentById_impl(const std::unordered_map<Guid, int> &idToIndexMap, const PoolAllocator<T> *allocator,
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

    template <typename T> T *addComponent_impl(PoolAllocator<T> *allocator, const YAML::Node &in)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        assert(allocator != nullptr);

        int componentGlobalIndex = (int)allocator->getCount();
        int componentType = ComponentType<T>::type;

        T *component = allocator->construct(mWorld, in);

        if (component != nullptr)
        {
            addIdToGlobalIndexMap_impl<T>(component->getId(), componentGlobalIndex, componentType);

            mIdState.mEntityIdToComponentIds[component->getEntityId()].push_back(
                std::make_pair(component->getId(), componentType));

            mIdState.mComponentIdsMarkedCreated.push_back(std::make_tuple(component->getEntityId(), component->getId(), componentType));
        }

        return component;
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

        T *component = allocator->construct(mWorld, Guid::newGuid());

        if (component != nullptr)
        {
            component->mEntityId = entityId;

            addIdToGlobalIndexMap_impl<T>(component->getId(), componentGlobalIndex, componentType);

            mIdState.mEntityIdToComponentIds[entityId].push_back(std::make_pair(component->getId(), componentType));

            mIdState.mComponentIdsMarkedCreated.push_back(std::make_tuple(entityId, component->getId(), componentType));
        }

        return component;
    }


    template <typename T> void addIdToGlobalIndexMap_impl(const Guid &id, int index, int type)
    {
        static_assert(std::is_base_of<Entity, T>() || std::is_base_of<Component, T>(),
                      "'T' is not of type Entity or Component");
    }

    template <> void addIdToGlobalIndexMap_impl<Entity>(const Guid &id, int index, int type)
    {
        mIdState.mEntityIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Transform>(const Guid &id, int index, int type)
    {
        mIdState.mTransformIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<MeshRenderer>(const Guid &id, int index, int type)
    {
        mIdState.mMeshRendererIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<SpriteRenderer>(const Guid &id, int index, int type)
    {
        mIdState.mSpriteRendererIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<LineRenderer>(const Guid &id, int index, int type)
    {
        mIdState.mLineRendererIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Rigidbody>(const Guid &id, int index, int type)
    {
        mIdState.mRigidbodyIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Camera>(const Guid &id, int index, int type)
    {
        mIdState.mCameraIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Light>(const Guid &id, int index, int type)
    {
        mIdState.mLightIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<SphereCollider>(const Guid &id, int index, int type)
    {
        mIdState.mSphereColliderIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<BoxCollider>(const Guid &id, int index, int type)
    {
        mIdState.mBoxColliderIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<CapsuleCollider>(const Guid &id, int index, int type)
    {
        mIdState.mCapsuleColliderIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<MeshCollider>(const Guid &id, int index, int type)
    {
        mIdState.mMeshColliderIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Terrain>(const Guid &id, int index, int type)
    {
        mIdState.mTerrainIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
};

template <typename T> struct SceneType
{
    static constexpr int type = PhysicsEngine::INVALID_TYPE;
};

template <typename> struct IsSceneInternal
{
    static constexpr bool value = false;
};

template <> struct IsSceneInternal<Scene>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif