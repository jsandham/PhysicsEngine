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
#include "../components/Transform.h"
#include "../components/Terrain.h"

namespace PhysicsEngine
{
// Simple structs used for grouping scene id to global index maps when passing to functions
struct SceneIdState
{
    // scene entity guid state
    std::unordered_map<Guid, int> mEntityGuidToGlobalIndex;

    // scene components guid state
    std::unordered_map<Guid, int> mTransformGuidToGlobalIndex;
    std::unordered_map<Guid, int> mMeshRendererGuidToGlobalIndex;
    std::unordered_map<Guid, int> mLineRendererGuidToGlobalIndex;
    std::unordered_map<Guid, int> mRigidbodyGuidToGlobalIndex;
    std::unordered_map<Guid, int> mCameraGuidToGlobalIndex;
    std::unordered_map<Guid, int> mLightGuidToGlobalIndex;
    std::unordered_map<Guid, int> mSphereColliderGuidToGlobalIndex;
    std::unordered_map<Guid, int> mBoxColliderGuidToGlobalIndex;
    std::unordered_map<Guid, int> mCapsuleColliderGuidToGlobalIndex;
    std::unordered_map<Guid, int> mMeshColliderGuidToGlobalIndex;
    std::unordered_map<Guid, int> mTerrainGuidToGlobalIndex;

    // scene guid state for all entity and components
    std::unordered_map<Guid, int> mGuidToGlobalIndex;
    std::unordered_map<Guid, int> mGuidToType;

    // entity guids to component guids
    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>> mEntityGuidToComponentIds;

    // entity creation/deletion state
    std::vector<Guid> mEntityGuidsMarkedCreated;
    std::vector<Guid> mEntityGuidsMarkedLatentDestroy;
    std::vector<std::pair<Guid, int>> mEntityGuidsMarkedMoved;

    // component create/deletion state
    std::vector<std::tuple<Guid, Guid, int>> mComponentGuidsMarkedCreated;
    std::vector<std::tuple<Guid, Guid, int>> mComponentGuidsMarkedLatentDestroy;
    std::vector<std::tuple<Guid, int, int>> mComponentGuidsMarkedMoved;

    // scene entity id state
    std::unordered_map<Id, int> mEntityIdToGlobalIndex;

    // scene components id state
    std::unordered_map<Id, int> mTransformIdToGlobalIndex;
    std::unordered_map<Id, int> mMeshRendererIdToGlobalIndex;
    std::unordered_map<Id, int> mLineRendererIdToGlobalIndex;
    std::unordered_map<Id, int> mRigidbodyIdToGlobalIndex;
    std::unordered_map<Id, int> mCameraIdToGlobalIndex;
    std::unordered_map<Id, int> mLightIdToGlobalIndex;
    std::unordered_map<Id, int> mSphereColliderIdToGlobalIndex;
    std::unordered_map<Id, int> mBoxColliderIdToGlobalIndex;
    std::unordered_map<Id, int> mCapsuleColliderIdToGlobalIndex;
    std::unordered_map<Id, int> mMeshColliderIdToGlobalIndex;
    std::unordered_map<Id, int> mTerrainIdToGlobalIndex;

    // scene id state for all entity and components
    std::unordered_map<Id, int> mIdToGlobalIndex;
    std::unordered_map<Id, int> mIdToType;

    std::unordered_map<Guid, Id> mGuidToId;
    std::unordered_map<Id, Guid> mIdToGuid;
};

// Simple structs used for grouping scene allocators when passing to functions
struct SceneAllocators
{
    // internal entity allocator
    PoolAllocator<Entity> mEntityAllocator;

    // internal component allocators
    PoolAllocator<Transform> mTransformAllocator;
    PoolAllocator<MeshRenderer> mMeshRendererAllocator;
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
    Scene(World *world, const Id &id);
    Scene(World *world, const Guid& guid, const Id& id);
    ~Scene();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    bool writeToYAML(const std::string &filepath) const;

    std::string getName() const;

    size_t getNumberOfEntities() const;
    size_t getNumberOfNonHiddenEntities() const;

    int getIndexOf(const Id &id) const;
    int getTypeOf(const Id &id) const;
    int getIndexOf(const Guid &guid) const;
    int getTypeOf(const Guid &guid) const;
    Entity *getEntityByIndex(size_t index) const;
    Entity *getEntityById(const Id &entityId) const;
    Entity *getEntityByGuid(const Guid &entityGuid) const;
    Component *getComponentById(const Id &componentId, int type) const;
    Component *getComponentByGuid(const Guid &componentGuid, int type) const;
    Component *addComponent(const Guid &entityGuid, int type);
    Component *addComponent(const YAML::Node &in, int type);

    Entity *createEntity();
    Entity *createEntity(const YAML::Node &in);
    Entity *createEntity(const std::string &name);
    Entity *createPrimitive(PrimitiveType type);
    Entity *createNonPrimitive(const Guid &meshGuid);
    Entity *createLight(LightType type);
    Entity *createCamera();

    void latentDestroyEntitiesInScene();
    void immediateDestroyEntitiesInScene();

    void latentDestroyEntity(const Guid &entityGuid);
    void immediateDestroyEntity(const Guid &entityGuid);
    void latentDestroyComponent(const Guid &entityGuid, const Guid &componentGuid, int componentType);
    void immediateDestroyComponent(const Guid &entityGuid, const Guid &componentGuid, int componentType);

    bool isMarkedForLatentDestroy(const Guid &guid);
    void clearIdsMarkedCreatedOrDestroyed();


    std::vector<std::pair<Guid, int>> getComponentsOnEntity(const Guid &entityGuid) const;

    std::vector<Guid> getEntityIdsMarkedCreated() const;
    std::vector<Guid> getEntityIdsMarkedLatentDestroy() const;
    std::vector<std::tuple<Guid, Guid, int>> getComponentIdsMarkedCreated() const;
    std::vector<std::tuple<Guid, Guid, int>> getComponentIdsMarkedLatentDestroy() const;

    template <typename T> size_t getNumberOfComponents() const;
    template <typename T> T* getComponentByIndex(size_t index) const;
    template <typename T> T* getComponentById(const Id& componentId) const;
    template <typename T> T* getComponentByGuid(const Guid &componentGuid) const;
    template <typename T> T* getComponent(const Guid& entityGuid) const;
    template <typename T> T* addComponent(const Guid& entityGuid);
    template <typename T> T* addComponent(const YAML::Node& in);

  private:
    void addToIdState(const Guid &guid, const Id &id, int index, int type);
    void removeFromIdState(const Guid &guid, const Id &id);

    template <typename T> void addToIdState_impl(const Guid &guid, const Id &id, int index, int type);
    template <typename T> void removeFromIdState_impl(const Guid &guid, const Id &id);
    template <typename T>
    T* getComponentById_impl(const std::unordered_map<Id, int>& idToIndexMap, const PoolAllocator<T>* allocator, const Id& id) const;
    template <typename T>
    T *getComponentByGuid_impl(const std::unordered_map<Guid, int> &guidToIndexMap, const PoolAllocator<T> *allocator,
                             const Guid &guid) const;
    template <typename T> T *getComponent_impl(const PoolAllocator<T> *allocator, const Guid &entityGuid) const;
    template <typename T> T* addComponent_impl(PoolAllocator<T>* allocator, const YAML::Node& in);
    template <typename T> T *addComponent_impl(PoolAllocator<T> *allocator, const Guid &entityGuid);
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