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

    template <typename T> size_t getNumberOfComponents() const;
    template <typename T> T* getComponentByIndex(size_t index) const;
    template <typename T> T* getComponentById(const Guid& componentId) const;
    template <typename T> T* getComponent(const Guid& entityId) const;
    template <typename T> T* addComponent(const Guid& entityId);
    template <typename T> T* addComponent(const YAML::Node& in);

  private:
    template <typename T>
    T* getComponentById_impl(const std::unordered_map<Guid, int>& idToIndexMap, const PoolAllocator<T>* allocator, const Guid& id) const;
    template <typename T> T* getComponent_impl(const PoolAllocator<T>* allocator, const Guid& entityId) const;
    template <typename T> T* addComponent_impl(PoolAllocator<T>* allocator, const YAML::Node& in);
    template <typename T> T* addComponent_impl(PoolAllocator<T>* allocator, const Guid& entityId);
    template <typename T> void addIdToGlobalIndexMap_impl(const Guid& id, int index, int type);
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