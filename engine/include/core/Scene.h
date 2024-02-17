#ifndef SCENE_H__
#define SCENE_H__

#include <string>
#include <unordered_map>

#include "Entity.h"
#include "Guid.h"
#include "PoolAllocator.h"
#include "WorldPrimitives.h"

#include "../components/BoxCollider.h"
#include "../components/Camera.h"
#include "../components/Light.h"
#include "../components/MeshRenderer.h"
#include "../components/Rigidbody.h"
#include "../components/SphereCollider.h"
#include "../components/Terrain.h"
#include "../components/Transform.h"

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
    std::unordered_map<Guid, int> mRigidbodyGuidToGlobalIndex;
    std::unordered_map<Guid, int> mCameraGuidToGlobalIndex;
    std::unordered_map<Guid, int> mLightGuidToGlobalIndex;
    std::unordered_map<Guid, int> mSphereColliderGuidToGlobalIndex;
    std::unordered_map<Guid, int> mBoxColliderGuidToGlobalIndex;
    std::unordered_map<Guid, int> mTerrainGuidToGlobalIndex;

    // scene guid state for all entity and components
    std::unordered_map<Guid, int> mGuidToGlobalIndex;
    std::unordered_map<Guid, int> mGuidToType;

    // entity guids to component guids
    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>> mEntityGuidToComponentIds;

    // entity creation/deletion state
    std::vector<Guid> mEntityGuidsMarkedCreated;
    std::vector<Guid> mEntityGuidsMarkedLatentDestroy;

    // component create/deletion state
    std::vector<std::tuple<Guid, Guid, int>> mComponentGuidsMarkedCreated;
    std::vector<std::tuple<Guid, Guid, int>> mComponentGuidsMarkedLatentDestroy;

    // scene entity id state
    std::unordered_map<Id, int> mEntityIdToGlobalIndex;

    // scene components id state
    std::unordered_map<Id, int> mTransformIdToGlobalIndex;
    std::unordered_map<Id, int> mMeshRendererIdToGlobalIndex;
    std::unordered_map<Id, int> mRigidbodyIdToGlobalIndex;
    std::unordered_map<Id, int> mCameraIdToGlobalIndex;
    std::unordered_map<Id, int> mLightIdToGlobalIndex;
    std::unordered_map<Id, int> mSphereColliderIdToGlobalIndex;
    std::unordered_map<Id, int> mBoxColliderIdToGlobalIndex;
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
    // entity allocator
    PoolAllocator<Entity> mEntityAllocator;

    // component allocators
    PoolAllocator<Transform> mTransformAllocator;
    PoolAllocator<MeshRenderer> mMeshRendererAllocator;
    PoolAllocator<Rigidbody> mRigidbodyAllocator;
    PoolAllocator<Camera> mCameraAllocator;
    PoolAllocator<Light> mLightAllocator;
    PoolAllocator<SphereCollider> mSphereColliderAllocator;
    PoolAllocator<BoxCollider> mBoxColliderAllocator;
    PoolAllocator<Terrain> mTerrainAllocator;

    PoolAllocator<TransformData> mTransformDataAllocator;
    PoolAllocator<SphereColliderData> mSphereColliderDataAllocator;
    PoolAllocator<BoxColliderData> mBoxColliderDataAllocator;
    PoolAllocator<RigidbodyData> mRigidbodyDataAllocator;


    PoolAllocator<size_t> mTransformIndicesAllocator;
};

class Scene
{
  private:
    Guid mGuid;
    Id mId;
    World *mWorld;

    // allocators for entities and components
    SceneAllocators mAllocators;

    // id state for entities and components
    SceneIdState mIdState;

    std::string mVersion;

  public:
    std::string mName;
    HideFlag mHide;

  public:
    Scene(World *world, const Id &id);
    Scene(World *world, const Guid &guid, const Id &id);

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getGuid() const;
    Id getId() const;

    bool writeToYAML(const std::string &filepath) const;

    size_t getNumberOfEntities() const;
    size_t getNumberOfNonHiddenEntities() const;

    int getIndexOf(const Id &id) const;
    int getTypeOf(const Id &id) const;
    int getIndexOf(const Guid &guid) const;
    int getTypeOf(const Guid &guid) const;
    Id getIdFromGuid(const Guid &guid) const;
    Guid getGuidFromId(const Id &id) const;
    Entity *getEntityByIndex(size_t index) const;
    Entity *getEntityById(const Id &entityId) const;
    Entity *getEntityByGuid(const Guid &entityGuid) const;

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

    size_t getTransformDataCount() const;
    TransformData *getTransformDataByIndex(size_t index) const;
    TransformData *getTransformDataFromTransformId(const Id &id) const;
    TransformData *getTransformDataFromTransformGuid(const Guid &guid) const;

    void setTransformPosition(const Id &id, const glm::vec3 &position);
    void setTransformRotation(const Id &id, const glm::quat &rotation);
    void setTransformScale(const Id &id, const glm::vec3 &scale);
    glm::vec3 getTransformPosition(const Id &id) const;
    glm::quat getTransformRotation(const Id &id) const;
    glm::vec3 getTransformScale(const Id &id) const;
    glm::mat4 getTransformModelMatrix(const Id& id) const;
    glm::vec3 getTransformForward(const Id &id) const;
    glm::vec3 getTransformUp(const Id &id) const;
    glm::vec3 getTransformRight(const Id &id) const;

    size_t getIndexOfTransformFromMeshRendererIndex(size_t index) const;
    TransformData* getTransformDataByMeshRendererIndex(size_t index) const;

    template <typename T> size_t getNumberOfComponents() const;
    template <typename T> T *getComponentByIndex(size_t index) const;
    template <typename T> T *getComponentById(const Id &componentId) const;
    template <typename T> T *getComponentByGuid(const Guid &componentGuid) const;
    template <typename T> T *getComponent(const Guid &entityGuid) const;
    template <typename T> T *addComponent(const Guid &entityGuid);
    template <typename T> T *addComponent(const YAML::Node &in);

  private:
    int getComponentIndex(const Guid &entityGuid, int componentType) const;

    void addEntityToIdState(int entityIndex, const Guid &entityGuid, const Id &entityId);
    void addComponentToIdState(std::unordered_map<Guid, int> &componentGuidToIndex,
                               std::unordered_map<Id, int> &componentIdToIndex, int componentIndex,
                               const Guid &entityGuid, const Guid &componentGuid, const Id &componentId,
                               int componentType);

    void removeEntityFromIdState(const Guid &entityGuid, const Id &entityId);
    void removeComponentFromIdState(std::unordered_map<Guid, int> &componentGuidToIndex,
                                    std::unordered_map<Id, int> &componentIdToIndex, const Guid &entityGuid,
                                    const Guid &componentGuid, const Id &componentId, int componentType);

    void moveEntityIndexInIdState(const Guid &entityGuid, const Id &entityId, int entityIndex);
    void moveComponentIndexInIdState(std::unordered_map<Guid, int> &componentGuidToIndex,
                                     std::unordered_map<Id, int> &componentIdToIndex, const Guid &componentGuid,
                                     const Id &componentId, int componentIndex);

    Entity *addEntity(const YAML::Node &in);
    Transform *addTransform(const YAML::Node &in);
    Rigidbody *addRigidbody(const YAML::Node &in);
    MeshRenderer *addMeshRenderer(const YAML::Node &in);
    Light *addLight(const YAML::Node &in);
    Camera *addCamera(const YAML::Node &in);
    SphereCollider *addSphereCollider(const YAML::Node &in);
    BoxCollider *addBoxCollider(const YAML::Node &in);
    Terrain *addTerrain(const YAML::Node &in);

    Entity *addEntity();
    Transform *addTransform(const Guid &entityGuid);
    Rigidbody *addRigidbody(const Guid &entityGuid);
    MeshRenderer *addMeshRenderer(const Guid &entityGuid);
    Light *addLight(const Guid &entityGuid);
    Camera *addCamera(const Guid &entityGuid);
    SphereCollider *addSphereCollider(const Guid &entityGuid);
    BoxCollider *addBoxCollider(const Guid &entityGuid);
    Terrain *addTerrain(const Guid &entityGuid);

    void removeEntity(const Guid &entityGuid);
    void removeTransform(const Guid &entityGuid, const Guid &componentGuid);
    void removeRigidbody(const Guid &entityGuid, const Guid &componentGuid);
    void removeMeshRenderer(const Guid &entityGuid, const Guid &componentGuid);
    void removeLight(const Guid &entityGuid, const Guid &componentGuid);
    void removeCamera(const Guid &entityGuid, const Guid &componentGuid);
    void removeSphereCollider(const Guid &entityGuid, const Guid &componentGuid);
    void removeBoxCollider(const Guid &entityGuid, const Guid &componentGuid);
    void removeTerrain(const Guid &entityGuid, const Guid &componentGuid);
};

template <typename T> struct SceneType
{
    static constexpr int type = PhysicsEngine::INVALID_TYPE;
};

} // namespace PhysicsEngine

#endif