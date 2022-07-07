#include <fstream>

#include "../../include/core/Scene.h"
#include "../../include/core/GLM.h"
#include "../../include/core/Types.h"
#include "../../include/core/Version.h"
#include "../../include/core/Entity.h"
#include "../../include/core/Log.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

template <> size_t Scene::getNumberOfComponents<Transform>() const
{
    return mAllocators.mTransformAllocator.getCount();
}

template <> size_t Scene::getNumberOfComponents<MeshRenderer>() const
{
    return mAllocators.mMeshRendererAllocator.getCount();
}

template <> size_t Scene::getNumberOfComponents<SpriteRenderer>() const
{
    return mAllocators.mSpriteRendererAllocator.getCount();
}

template <> size_t Scene::getNumberOfComponents<LineRenderer>() const
{
    return mAllocators.mLineRendererAllocator.getCount();
}

template <> size_t Scene::getNumberOfComponents<Rigidbody>() const
{
    return mAllocators.mRigidbodyAllocator.getCount();
}

template <> size_t Scene::getNumberOfComponents<Camera>() const
{
    return mAllocators.mCameraAllocator.getCount();
}

template <> size_t Scene::getNumberOfComponents<Light>() const
{
    return mAllocators.mLightAllocator.getCount();
}

template <> size_t Scene::getNumberOfComponents<SphereCollider>() const
{
    return mAllocators.mSphereColliderAllocator.getCount();
}

template <> size_t Scene::getNumberOfComponents<BoxCollider>() const
{
    return mAllocators.mBoxColliderAllocator.getCount();
}

template <> size_t Scene::getNumberOfComponents<CapsuleCollider>() const
{
    return mAllocators.mCapsuleColliderAllocator.getCount();
}

template <> size_t Scene::getNumberOfComponents<MeshCollider>() const
{
    return mAllocators.mMeshColliderAllocator.getCount();
}

template <> size_t Scene::getNumberOfComponents<Terrain>() const
{
    return mAllocators.mTerrainAllocator.getCount();
}

template <> Transform* Scene::getComponentByIndex<Transform>(size_t index) const
{
    return mAllocators.mTransformAllocator.get(index);
}

template <> MeshRenderer* Scene::getComponentByIndex<MeshRenderer>(size_t index) const
{
    return mAllocators.mMeshRendererAllocator.get(index);
}

template <> SpriteRenderer* Scene::getComponentByIndex<SpriteRenderer>(size_t index) const
{
    return mAllocators.mSpriteRendererAllocator.get(index);
}

template <> LineRenderer* Scene::getComponentByIndex<LineRenderer>(size_t index) const
{
    return mAllocators.mLineRendererAllocator.get(index);
}

template <> Rigidbody* Scene::getComponentByIndex<Rigidbody>(size_t index) const
{
    return mAllocators.mRigidbodyAllocator.get(index);
}

template <> Camera* Scene::getComponentByIndex<Camera>(size_t index) const
{
    return mAllocators.mCameraAllocator.get(index);
}

template <> Light* Scene::getComponentByIndex<Light>(size_t index) const
{
    return mAllocators.mLightAllocator.get(index);
}

template <> SphereCollider* Scene::getComponentByIndex<SphereCollider>(size_t index) const
{
    return mAllocators.mSphereColliderAllocator.get(index);
}

template <> BoxCollider* Scene::getComponentByIndex<BoxCollider>(size_t index) const
{
    return mAllocators.mBoxColliderAllocator.get(index);
}

template <> CapsuleCollider* Scene::getComponentByIndex<CapsuleCollider>(size_t index) const
{
    return mAllocators.mCapsuleColliderAllocator.get(index);
}

template <> MeshCollider* Scene::getComponentByIndex<MeshCollider>(size_t index) const
{
    return mAllocators.mMeshColliderAllocator.get(index);
}

template <> Terrain* Scene::getComponentByIndex<Terrain>(size_t index) const
{
    return mAllocators.mTerrainAllocator.get(index);
}

template <> Transform* Scene::getComponentById<Transform>(const Guid& componentId) const
{
    return getComponentById_impl(mIdState.mTransformIdToGlobalIndex, &mAllocators.mTransformAllocator, componentId);
}

template <> MeshRenderer* Scene::getComponentById<MeshRenderer>(const Guid& componentId) const
{
    return getComponentById_impl(mIdState.mMeshRendererIdToGlobalIndex, &mAllocators.mMeshRendererAllocator,
        componentId);
}

template <> SpriteRenderer* Scene::getComponentById<SpriteRenderer>(const Guid& componentId) const
{
    return getComponentById_impl(mIdState.mSpriteRendererIdToGlobalIndex, &mAllocators.mSpriteRendererAllocator,
        componentId);
}

template <> LineRenderer* Scene::getComponentById<LineRenderer>(const Guid& componentId) const
{
    return getComponentById_impl(mIdState.mLineRendererIdToGlobalIndex, &mAllocators.mLineRendererAllocator,
        componentId);
}

template <> Rigidbody* Scene::getComponentById<Rigidbody>(const Guid& componentId) const
{
    return getComponentById_impl(mIdState.mRigidbodyIdToGlobalIndex, &mAllocators.mRigidbodyAllocator, componentId);
}

template <> Camera* Scene::getComponentById<Camera>(const Guid& componentId) const
{
    return getComponentById_impl(mIdState.mCameraIdToGlobalIndex, &mAllocators.mCameraAllocator, componentId);
}

template <> Light* Scene::getComponentById<Light>(const Guid& componentId) const
{
    return getComponentById_impl(mIdState.mLightIdToGlobalIndex, &mAllocators.mLightAllocator, componentId);
}

template <> SphereCollider* Scene::getComponentById<SphereCollider>(const Guid& componentId) const
{
    return getComponentById_impl(mIdState.mSphereColliderIdToGlobalIndex, &mAllocators.mSphereColliderAllocator,
        componentId);
}

template <> BoxCollider* Scene::getComponentById<BoxCollider>(const Guid& componentId) const
{
    return getComponentById_impl(mIdState.mBoxColliderIdToGlobalIndex, &mAllocators.mBoxColliderAllocator,
        componentId);
}

template <> CapsuleCollider* Scene::getComponentById<CapsuleCollider>(const Guid& componentId) const
{
    return getComponentById_impl(mIdState.mCapsuleColliderIdToGlobalIndex, &mAllocators.mCapsuleColliderAllocator,
        componentId);
}

template <> MeshCollider* Scene::getComponentById<MeshCollider>(const Guid& componentId) const
{
    return getComponentById_impl(mIdState.mMeshColliderIdToGlobalIndex, &mAllocators.mMeshColliderAllocator,
        componentId);
}

template <> Terrain* Scene::getComponentById<Terrain>(const Guid& componentId) const
{
    return getComponentById_impl(mIdState.mTerrainIdToGlobalIndex, &mAllocators.mTerrainAllocator, componentId);
}

template <> Transform* Scene::getComponent<Transform>(const Guid& entityId) const
{
    // Transform occurs at same index as its entity since all entities have a transform
    return getComponentByIndex<Transform>(getIndexOf(entityId));
}

template <> MeshRenderer* Scene::getComponent<MeshRenderer>(const Guid& entityId) const
{
    return getComponent_impl(&mAllocators.mMeshRendererAllocator, entityId);
}

template <> SpriteRenderer* Scene::getComponent<SpriteRenderer>(const Guid& entityId) const
{
    return getComponent_impl(&mAllocators.mSpriteRendererAllocator, entityId);
}

template <> LineRenderer* Scene::getComponent<LineRenderer>(const Guid& entityId) const
{
    return getComponent_impl(&mAllocators.mLineRendererAllocator, entityId);
}

template <> Rigidbody* Scene::getComponent<Rigidbody>(const Guid& entityId) const
{
    return getComponent_impl(&mAllocators.mRigidbodyAllocator, entityId);
}

template <> Camera* Scene::getComponent<Camera>(const Guid& entityId) const
{
    return getComponent_impl(&mAllocators.mCameraAllocator, entityId);
}

template <> Light* Scene::getComponent<Light>(const Guid& entityId) const
{
    return getComponent_impl(&mAllocators.mLightAllocator, entityId);
}

template <> SphereCollider* Scene::getComponent<SphereCollider>(const Guid& entityId) const
{
    return getComponent_impl(&mAllocators.mSphereColliderAllocator, entityId);
}

template <> BoxCollider* Scene::getComponent<BoxCollider>(const Guid& entityId) const
{
    return getComponent_impl(&mAllocators.mBoxColliderAllocator, entityId);
}

template <> CapsuleCollider* Scene::getComponent<CapsuleCollider>(const Guid& entityId) const
{
    return getComponent_impl(&mAllocators.mCapsuleColliderAllocator, entityId);
}

template <> MeshCollider* Scene::getComponent<MeshCollider>(const Guid& entityId) const
{
    return getComponent_impl(&mAllocators.mMeshColliderAllocator, entityId);
}

template <> Terrain* Scene::getComponent<Terrain>(const Guid& entityId) const
{
    return getComponent_impl(&mAllocators.mTerrainAllocator, entityId);
}

template <> MeshRenderer* Scene::addComponent<MeshRenderer>(const Guid& entityId)
{
    return addComponent_impl(&mAllocators.mMeshRendererAllocator, entityId);
}

template <> SpriteRenderer* Scene::addComponent<SpriteRenderer>(const Guid& entityId)
{
    return addComponent_impl(&mAllocators.mSpriteRendererAllocator, entityId);
}

template <> LineRenderer* Scene::addComponent<LineRenderer>(const Guid& entityId)
{
    return addComponent_impl(&mAllocators.mLineRendererAllocator, entityId);
}

template <> Rigidbody* Scene::addComponent<Rigidbody>(const Guid& entityId)
{
    return addComponent_impl(&mAllocators.mRigidbodyAllocator, entityId);
}

template <> Camera* Scene::addComponent<Camera>(const Guid& entityId)
{
    return addComponent_impl(&mAllocators.mCameraAllocator, entityId);
}

template <> Light* Scene::addComponent<Light>(const Guid& entityId)
{
    return addComponent_impl(&mAllocators.mLightAllocator, entityId);
}

template <> SphereCollider* Scene::addComponent<SphereCollider>(const Guid& entityId)
{
    return addComponent_impl(&mAllocators.mSphereColliderAllocator, entityId);
}

template <> BoxCollider* Scene::addComponent<BoxCollider>(const Guid& entityId)
{
    return addComponent_impl(&mAllocators.mBoxColliderAllocator, entityId);
}

template <> CapsuleCollider* Scene::addComponent<CapsuleCollider>(const Guid& entityId)
{
    return addComponent_impl(&mAllocators.mCapsuleColliderAllocator, entityId);
}

template <> MeshCollider* Scene::addComponent<MeshCollider>(const Guid& entityId)
{
    return addComponent_impl(&mAllocators.mMeshColliderAllocator, entityId);
}

template <> Terrain* Scene::addComponent<Terrain>(const Guid& entityId)
{
    return addComponent_impl(&mAllocators.mTerrainAllocator, entityId);
}

template <> Transform* Scene::addComponent<Transform>(const YAML::Node& in)
{
    return addComponent_impl(&mAllocators.mTransformAllocator, in);
}

template <> MeshRenderer* Scene::addComponent<MeshRenderer>(const YAML::Node& in)
{
    return addComponent_impl(&mAllocators.mMeshRendererAllocator, in);
}

template <> SpriteRenderer* Scene::addComponent<SpriteRenderer>(const YAML::Node& in)
{
    return addComponent_impl(&mAllocators.mSpriteRendererAllocator, in);
}

template <> LineRenderer* Scene::addComponent<LineRenderer>(const YAML::Node& in)
{
    return addComponent_impl(&mAllocators.mLineRendererAllocator, in);
}

template <> Rigidbody* Scene::addComponent<Rigidbody>(const YAML::Node& in)
{
    return addComponent_impl(&mAllocators.mRigidbodyAllocator, in);
}

template <> Camera* Scene::addComponent<Camera>(const YAML::Node& in)
{
    return addComponent_impl(&mAllocators.mCameraAllocator, in);
}

template <> Light* Scene::addComponent<Light>(const YAML::Node& in)
{
    return addComponent_impl(&mAllocators.mLightAllocator, in);
}

template <> SphereCollider* Scene::addComponent<SphereCollider>(const YAML::Node& in)
{
    return addComponent_impl(&mAllocators.mSphereColliderAllocator, in);
}

template <> BoxCollider* Scene::addComponent<BoxCollider>(const YAML::Node& in)
{
    return addComponent_impl(&mAllocators.mBoxColliderAllocator, in);
}

template <> CapsuleCollider* Scene::addComponent<CapsuleCollider>(const YAML::Node& in)
{
    return addComponent_impl(&mAllocators.mCapsuleColliderAllocator, in);
}

template <> MeshCollider* Scene::addComponent<MeshCollider>(const YAML::Node& in)
{
    return addComponent_impl(&mAllocators.mMeshColliderAllocator, in);
}

template <> Terrain* Scene::addComponent<Terrain>(const YAML::Node& in)
{
    return addComponent_impl(&mAllocators.mTerrainAllocator, in);
}

template <> void Scene::addIdToGlobalIndexMap_impl<Entity>(const Guid& id, int index, int type)
{
    mIdState.mEntityIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void Scene::addIdToGlobalIndexMap_impl<Transform>(const Guid& id, int index, int type)
{
    mIdState.mTransformIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void Scene::addIdToGlobalIndexMap_impl<MeshRenderer>(const Guid& id, int index, int type)
{
    mIdState.mMeshRendererIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void Scene::addIdToGlobalIndexMap_impl<SpriteRenderer>(const Guid& id, int index, int type)
{
    mIdState.mSpriteRendererIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void Scene::addIdToGlobalIndexMap_impl<LineRenderer>(const Guid& id, int index, int type)
{
    mIdState.mLineRendererIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void Scene::addIdToGlobalIndexMap_impl<Rigidbody>(const Guid& id, int index, int type)
{
    mIdState.mRigidbodyIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void Scene::addIdToGlobalIndexMap_impl<Camera>(const Guid& id, int index, int type)
{
    mIdState.mCameraIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void Scene::addIdToGlobalIndexMap_impl<Light>(const Guid& id, int index, int type)
{
    mIdState.mLightIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void Scene::addIdToGlobalIndexMap_impl<SphereCollider>(const Guid& id, int index, int type)
{
    mIdState.mSphereColliderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void Scene::addIdToGlobalIndexMap_impl<BoxCollider>(const Guid& id, int index, int type)
{
    mIdState.mBoxColliderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void Scene::addIdToGlobalIndexMap_impl<CapsuleCollider>(const Guid& id, int index, int type)
{
    mIdState.mCapsuleColliderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void Scene::addIdToGlobalIndexMap_impl<MeshCollider>(const Guid& id, int index, int type)
{
    mIdState.mMeshColliderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void Scene::addIdToGlobalIndexMap_impl<Terrain>(const Guid& id, int index, int type)
{
    mIdState.mTerrainIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <typename T>
T* Scene::getComponentById_impl(const std::unordered_map<Guid, int>& idToIndexMap, const PoolAllocator<T>* allocator, const Guid& id) const
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

template <typename T> T* Scene::getComponent_impl(const PoolAllocator<T>* allocator, const Guid& entityId) const
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

template <typename T> T* Scene::addComponent_impl(PoolAllocator<T>* allocator, const YAML::Node& in)
{
    static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

    assert(allocator != nullptr);

    int componentGlobalIndex = (int)allocator->getCount();
    int componentType = ComponentType<T>::type;

    T* component = allocator->construct(mWorld, in);

    if (component != nullptr)
    {
        addIdToGlobalIndexMap_impl<T>(component->getId(), componentGlobalIndex, componentType);

        mIdState.mEntityIdToComponentIds[component->getEntityId()].push_back(
            std::make_pair(component->getId(), componentType));

        mIdState.mComponentIdsMarkedCreated.push_back(std::make_tuple(component->getEntityId(), component->getId(), componentType));
    }

    return component;
}

template <typename T> T* Scene::addComponent_impl(PoolAllocator<T>* allocator, const Guid& entityId)
{
    static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

    assert(allocator != nullptr);

    if (getTypeOf(entityId) != EntityType<Entity>::type)
    {
        return nullptr;
    }

    int componentGlobalIndex = (int)allocator->getCount();
    int componentType = ComponentType<T>::type;

    T* component = allocator->construct(mWorld, Guid::newGuid());

    if (component != nullptr)
    {
        component->mEntityId = entityId;

        addIdToGlobalIndexMap_impl<T>(component->getId(), componentGlobalIndex, componentType);

        mIdState.mEntityIdToComponentIds[entityId].push_back(std::make_pair(component->getId(), componentType));

        mIdState.mComponentIdsMarkedCreated.push_back(std::make_tuple(entityId, component->getId(), componentType));
    }

    return component;
}


Scene::Scene(World *world) : Object(world)
{
    mName = "Unnamed scene";
    mVersion = SCENE_VERSION;
}

Scene::Scene(World *world, const Guid& id) : Object(world, id)
{
    mName = "Unnamed scene";
    mVersion = SCENE_VERSION;
}

Scene::~Scene()
{
}

void Scene::serialize(YAML::Node &out) const
{
    Object::serialize(out);

    out["name"] = mName;
    out["version"] = mVersion;
}

void Scene::deserialize(const YAML::Node &in)
{
    Object::deserialize(in);

    mName = YAML::getValue<std::string>(in, "name");
    mVersion = YAML::getValue<std::string>(in, "version");

    assert(getId() == YAML::getValue<Guid>(in, "id"));

    // load all objects found in scene file
    for (YAML::const_iterator it = in.begin(); it != in.end(); ++it)
    {
        if (it->first.IsScalar() && it->second.IsMap())
        {
            int objectType = YAML::getValue<int>(it->second, "type");
            Guid objectId = YAML::getValue<Guid>(it->second, "id");
            HideFlag objectHideFlag = YAML::getValue<HideFlag>(it->second, "hide");

            if (objectHideFlag != HideFlag::DontSave)
            {
                if (isEntity(objectType))
                {
                    Entity *entity = getEntityById(objectId);
                    if (entity == nullptr)
                    {
                        entity = createEntity(it->second);
                    }

                    if (entity != nullptr)
                    {
                        entity->deserialize(it->second);
                    }
                    else
                    {
                        Log::warn("A scene entity could not be loaded from scene file. Skipping it.\n");
                    }
                }
                else if (isComponent(objectType))
                {
                    Guid entityId = YAML::getValue<Guid>(it->second, "entityId");

                    Component *component = getComponentById(objectId, objectType);
                    if (component == nullptr)
                    {
                        component = addComponent(it->second, objectType);
                    }

                    if (component != nullptr)
                    {
                        component->deserialize(it->second);
                    }
                    else
                    {
                        Log::warn("A scene component could not be loaded from scene file. Skipping it.\n");
                    }
                }
            }
        }
    }
}

int Scene::getType() const
{
    return PhysicsEngine::SCENE_TYPE;
}

std::string Scene::getObjectName() const
{
    return PhysicsEngine::SCENE_NAME;
}

bool Scene::writeToYAML(const std::string &filepath) const
{
    std::ofstream out;
    out.open(filepath);

    if (!out.is_open())
    {
        std::string errorMessage = "Failed to open scene file " + filepath + "\n";
        Log::error(&errorMessage[0]);
        return false;
    }

    YAML::Node sceneNode;
    serialize(sceneNode);

    out << sceneNode;
    out << "\n";

    for (size_t i = 0; i < getNumberOfEntities(); i++)
    {
        const Entity *entity = getEntityByIndex(i);

        if (entity->mHide == HideFlag::None)
        {
            YAML::Node en;
            entity->serialize(en);

            YAML::Node entityNode;
            entityNode[entity->getObjectName()] = en;

            out << entityNode;
            out << "\n";

            std::vector<std::pair<Guid, int>> temp = entity->getComponentsOnEntity();
            for (size_t j = 0; j < temp.size(); j++)
            {
                //Component *component = nullptr;
                //Component *getInternalComponent(const SceneAllocators &allocators, const SceneIdState &state,
                //                                const Guid &id, int type);
                //if (Component::isInternal(temp[j].second))
                //{
                //    component =
                //        PhysicsEngine::getInternalComponent(mAllocators, mIdState, temp[j].first, temp[j].second);
                //}

                Component *component = getComponentById(temp[j].first, temp[j].second);

                if (component->mHide == HideFlag::None)
                {
                    YAML::Node cn;
                    component->serialize(cn);

                    YAML::Node componentNode;
                    componentNode[component->getObjectName()] = cn;

                    out << componentNode;
                    out << "\n";
                }
            }
        }
    }

    out.close();

    return true;
}

void Scene::load(const std::string &filepath)
{
    if (filepath.empty())
    {
        return;
    }

    /*YAML::Node in = YAML::LoadFile(filepath);

    if (!in.IsMap()) {
        return false;
    }

    mId = YAML::getValue<Guid>(in, "id");

    for (YAML::const_iterator it = in.begin(); it != in.end(); ++it) {
        if (it->first.IsScalar() && it->second.IsMap()) {
            if (loadSceneObjectFromYAML(it->second) == nullptr) {
                return false;
            }
        }
    }

    return true;*/
}

std::string Scene::getName() const
{
    return mName;
}

size_t Scene::getNumberOfEntities() const
{
    return mAllocators.mEntityAllocator.getCount();
}

size_t Scene::getNumberOfNonHiddenEntities() const
{
    size_t count = 0;
    for (size_t i = 0; i < getNumberOfEntities(); i++)
    {
        const Entity *entity = getEntityByIndex(i);
        if (entity->mHide == HideFlag::None)
        {
            count++;
        }
    }

    return count;
}

Entity *Scene::getEntityByIndex(size_t index) const
{
    return mAllocators.mEntityAllocator.get(index);
}

Entity *Scene::getEntityById(const Guid &entityId) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mEntityIdToGlobalIndex.find(entityId);
    if (it != mIdState.mEntityIdToGlobalIndex.end())
    {
        return mAllocators.mEntityAllocator.get(it->second);
    }
    else
    {
        return nullptr;
    }
}

Component *Scene::getComponentById(const Guid &componentId, int type) const
{
    switch (type)
    {
    case ComponentType<Transform>::type: {
        return getComponentById<Transform>(componentId);
    }
    case ComponentType<Rigidbody>::type: {
        return getComponentById<Rigidbody>(componentId);
    }
    case ComponentType<Camera>::type: {
        return getComponentById<Camera>(componentId);
    }
    case ComponentType<MeshRenderer>::type: {
        return getComponentById<MeshRenderer>(componentId);
    }
    case ComponentType<SpriteRenderer>::type: {
        return getComponentById<SpriteRenderer>(componentId);
    }
    case ComponentType<LineRenderer>::type: {
        return getComponentById<LineRenderer>(componentId);
    }
    case ComponentType<Light>::type: {
        return getComponentById<Light>(componentId);
    }
    case ComponentType<BoxCollider>::type: {
        return getComponentById<BoxCollider>(componentId);
    }
    case ComponentType<SphereCollider>::type: {
        return getComponentById<SphereCollider>(componentId);
    }
    case ComponentType<MeshCollider>::type: {
        return getComponentById<MeshCollider>(componentId);
    }
    case ComponentType<CapsuleCollider>::type: {
        return getComponentById<CapsuleCollider>(componentId);
    }
    case ComponentType<Terrain>::type: {
        return getComponentById<Terrain>(componentId);
    }
    }

    return nullptr;
}

Component *Scene::addComponent(const YAML::Node &in, int type)
{
    switch (type)
    {
    case ComponentType<Transform>::type: {
        return addComponent<Transform>(in);
    }
    case ComponentType<Rigidbody>::type: {
        return addComponent<Rigidbody>(in);
    }
    case ComponentType<Camera>::type: {
        return addComponent<Camera>(in);
    }
    case ComponentType<MeshRenderer>::type: {
        return addComponent<MeshRenderer>(in);
    }
    case ComponentType<SpriteRenderer>::type: {
        return addComponent<SpriteRenderer>(in);
    }
    case ComponentType<LineRenderer>::type: {
        return addComponent<LineRenderer>(in);
    }
    case ComponentType<Light>::type: {
        return addComponent<Light>(in);
    }
    case ComponentType<BoxCollider>::type: {
        return addComponent<BoxCollider>(in);
    }
    case ComponentType<SphereCollider>::type: {
        return addComponent<SphereCollider>(in);
    }
    case ComponentType<MeshCollider>::type: {
        return addComponent<MeshCollider>(in);
    }
    case ComponentType<CapsuleCollider>::type: {
        return addComponent<CapsuleCollider>(in);
    }
    case ComponentType<Terrain>::type: {
        return addComponent<Terrain>(in);
    }
    }

    return nullptr;
}

int Scene::getIndexOf(const Guid &id) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mIdToGlobalIndex.find(id);
    if (it != mIdState.mIdToGlobalIndex.end())
    {
        return it->second;
    }

    return -1;
}

int Scene::getTypeOf(const Guid &id) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mIdToType.find(id);
    if (it != mIdState.mIdToType.end())
    {
        return it->second;
    }

    return -1;
}

void Scene::latentDestroyEntitiesInScene()
{
    // latent destroy all entities (and thereby also all components)
    for (size_t i = 0; i < getNumberOfEntities(); i++)
    {
        Entity *entity = getEntityByIndex(i);

        if (!entity->mDoNotDestroy)
        {
            latentDestroyEntity(entity->getId());
        }
    }
}

void Scene::immediateDestroyEntitiesInScene()
{
    // immediate destroy all entities (and thereby also all components)
    std::vector<Guid> entitiesToDestroy;
    for (size_t i = 0; i < getNumberOfEntities(); i++)
    {
        Entity *entity = getEntityByIndex(i);

        if (!entity->mDoNotDestroy)
        {
            entitiesToDestroy.push_back(entity->getId());
        }
    }

    for (size_t i = 0; i < entitiesToDestroy.size(); i++)
    {
        Log::info(("Immediate destroy entity with id: " + entitiesToDestroy[i].toString() + "\n").c_str());
        immediateDestroyEntity(entitiesToDestroy[i]);
    }
}

Entity *Scene::createEntity()
{
    int globalIndex = (int)mAllocators.mEntityAllocator.getCount();
    int type = EntityType<Entity>::type;

    Entity *entity = mAllocators.mEntityAllocator.construct(mWorld, Guid::newGuid());

    if (entity != nullptr)
    {
        addIdToGlobalIndexMap_impl<Entity>(entity->getId(), globalIndex, type);

        mIdState.mEntityIdToComponentIds[entity->getId()] = std::vector<std::pair<Guid, int>>();

        mIdState.mEntityIdsMarkedCreated.push_back(entity->getId());
    }

    // Add transform (all entities must have a transform)
    int componentGlobalIndex = (int)mAllocators.mTransformAllocator.getCount();
    int componentType = ComponentType<Transform>::type;

    Transform *component = mAllocators.mTransformAllocator.construct(mWorld, Guid::newGuid());

    assert(component != nullptr);

    component->mEntityId = entity->getId();

    addIdToGlobalIndexMap_impl<Transform>(component->getId(), componentGlobalIndex, componentType);

    mIdState.mEntityIdToComponentIds[entity->getId()].push_back(std::make_pair(component->getId(), componentType));

    mIdState.mComponentIdsMarkedCreated.push_back(std::make_tuple(entity->getId(), component->getId(), componentType));

    return entity;
}

Entity *Scene::createEntity(const YAML::Node &in) // currently create entity without transform...pass in transform node as well??
{
    int globalIndex = (int)mAllocators.mEntityAllocator.getCount();
    int type = EntityType<Entity>::type;

    Entity *entity = mAllocators.mEntityAllocator.construct(mWorld, in);

    if (entity != nullptr)
    {
        addIdToGlobalIndexMap_impl<Entity>(entity->getId(), globalIndex, type);

        mIdState.mEntityIdToComponentIds[entity->getId()] = std::vector<std::pair<Guid, int>>();

        mIdState.mEntityIdsMarkedCreated.push_back(entity->getId());
    }

    //// Add transform (all entities must have a transform)
    //int componentGlobalIndex = (int)mAllocators.mTransformAllocator.getCount();
    //int componentType = ComponentType<Transform>::type;

    //Transform *component = mAllocators.mTransformAllocator.construct(mWorld, Guid::newGuid());

    //assert(component != nullptr);

    //component->mEntityId = entity->getId();

    //addIdToGlobalIndexMap_impl<Transform>(component->getId(), componentGlobalIndex, componentType);

    //mIdState.mEntityIdToComponentIds[entity->getId()].push_back(std::make_pair(component->getId(), componentType));

    //mIdState.mComponentIdsMarkedCreated.push_back(std::make_tuple(entity->getId(), component->getId(), componentType));

    return entity;
}

Entity *Scene::createEntity(const std::string &name)
{
    Entity *entity = createEntity();
    if (entity != nullptr)
    {
        entity->setName(name);
        return entity;
    }

    return nullptr;
}

Entity *Scene::createPrimitive(PrimitiveType type)
{
    Mesh *mesh = mWorld->getPrimtiveMesh(type);
    Entity *entity = createEntity();
    assert(mesh != nullptr);
    assert(entity != nullptr);

    MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>();
    assert(meshRenderer != nullptr);

    Transform *transform = entity->getComponent<Transform>();
    assert(transform != nullptr);

    entity->setName(mesh->getName());

    transform->setPosition(glm::vec3(0, 0, 0));
    transform->setRotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    transform->setScale(glm::vec3(1, 1, 1));
    meshRenderer->setMesh(mesh->getId());
    meshRenderer->setMaterial(mWorld->getPrimtiveMaterial()->getId());

    return entity;
}

Entity *Scene::createNonPrimitive(const Guid &meshId)
{
    Mesh *mesh = mWorld->getAssetById<Mesh>(meshId);
    if (mesh == nullptr)
    {
        return nullptr;
    }

    Entity *entity = createEntity();
    assert(entity != nullptr);

    MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>();
    assert(meshRenderer != nullptr);

    Transform *transform = entity->getComponent<Transform>();
    assert(transform != nullptr);

    entity->setName(mesh->getName());

    transform->setPosition(glm::vec3(0, 0, 0));
    transform->setRotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    transform->setScale(glm::vec3(1, 1, 1));
    meshRenderer->setMesh(meshId);
    meshRenderer->setMaterial(mWorld->getPrimtiveMaterial()->getId());

    return entity;
}

Entity *Scene::createLight(LightType type)
{
    Entity *entity = createEntity();
    Light *light = entity->addComponent<Light>();

    switch (type)
    {
    case LightType::Directional:
        light->mLightType = LightType::Directional;
        break;
    case LightType::Spot:
        light->mLightType = LightType::Spot;
        break;
    case LightType::Point:
        light->mLightType = LightType::Point;
        break;
    }

    return entity;
}

Entity *Scene::createCamera()
{
    Entity *entity = createEntity();
    entity->addComponent<Camera>();

    return entity;
}

void Scene::latentDestroyEntity(const Guid &entityId)
{
    mIdState.mEntityIdsMarkedLatentDestroy.push_back(entityId);

    // add any components found on the entity to the latent destroy component list
    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::const_iterator it =
        mIdState.mEntityIdToComponentIds.find(entityId);

    assert(it != mIdState.mEntityIdToComponentIds.end());

    for (size_t i = 0; i < it->second.size(); i++)
    {
        latentDestroyComponent(entityId, it->second[i].first, it->second[i].second);
    }
}

void Scene::immediateDestroyEntity(const Guid &entityId)
{
    // Destroy components on entity
    std::vector<std::pair<Guid, int>> componentsOnEntity = mIdState.mEntityIdToComponentIds[entityId];
    for (size_t i = 0; i < componentsOnEntity.size(); i++)
    {
        immediateDestroyComponent(entityId, componentsOnEntity[i].first, componentsOnEntity[i].second);
    }

    assert(mIdState.mEntityIdToComponentIds[entityId].size() == 0);

    mIdState.mEntityIdToComponentIds.erase(entityId);

    // Destroy entity
    int index = getIndexOf(entityId);
    Entity *swap = mAllocators.mEntityAllocator.destruct(index);
    
    mIdState.mEntityIdToGlobalIndex.erase(entityId);
    mIdState.mIdToGlobalIndex.erase(entityId);
    mIdState.mIdToType.erase(entityId);
    
    if (swap != nullptr)
    {
        mIdState.mEntityIdToGlobalIndex[swap->getId()] = index;
        mIdState.mIdToGlobalIndex[swap->getId()] = index;
        mIdState.mIdToType[swap->getId()] = EntityType<Entity>::type;
    }
}

void Scene::latentDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType)
{
    mIdState.mComponentIdsMarkedLatentDestroy.push_back(std::make_tuple(entityId, componentId, componentType));
}

void Scene::immediateDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType)
{
    // remove from entity component list
    std::vector<std::pair<Guid, int>> &componentsOnEntity = mIdState.mEntityIdToComponentIds[entityId];

    std::vector<std::pair<Guid, int>>::iterator it = componentsOnEntity.begin();
    while (it < componentsOnEntity.end())
    {
        if (it->second == componentType && it->first == componentId)
        {
            break;
        }

        it++;
    }

    if (it < componentsOnEntity.end())
    {
        componentsOnEntity.erase(it);
    }




   /* if (Component::isInternal(componentType))
    {
        destroyInternalComponent(mAllocators, mIdState, entityId, componentId, componentType, getIndexOf(componentId));
    }*/

    // Destroy component
    int index = getIndexOf(componentId);
    Component *swap = nullptr;
    
    if (componentType == ComponentType<Transform>::type)
    {
        swap = mAllocators.mTransformAllocator.destruct(index);
    
        mIdState.mTransformIdToGlobalIndex.erase(componentId);
        mIdState.mIdToGlobalIndex.erase(componentId);
        mIdState.mIdToType.erase(componentId);
    
        if (swap != nullptr)
        {
            mIdState.mTransformIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = componentType;
        }
    }
    else if (componentType == ComponentType<Rigidbody>::type)
    {
        swap = mAllocators.mRigidbodyAllocator.destruct(index);
    
        mIdState.mRigidbodyIdToGlobalIndex.erase(componentId);
        mIdState.mIdToGlobalIndex.erase(componentId);
        mIdState.mIdToType.erase(componentId);
    
        if (swap != nullptr)
        {
            mIdState.mRigidbodyIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = componentType;
        }
    }
    else if (componentType == ComponentType<Camera>::type)
    {
        swap = mAllocators.mCameraAllocator.destruct(index);
    
        mIdState.mCameraIdToGlobalIndex.erase(componentId);
        mIdState.mIdToGlobalIndex.erase(componentId);
        mIdState.mIdToType.erase(componentId);
    
        if (swap != nullptr)
        {
            mIdState.mCameraIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = componentType;
        }
    }
    else if (componentType == ComponentType<MeshRenderer>::type)
    {
        swap = mAllocators.mMeshRendererAllocator.destruct(index);
    
        mIdState.mMeshRendererIdToGlobalIndex.erase(componentId);
        mIdState.mIdToGlobalIndex.erase(componentId);
        mIdState.mIdToType.erase(componentId);
    
        if (swap != nullptr)
        {
            mIdState.mMeshRendererIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = componentType;
        }
    }
    else if (componentType == ComponentType<SpriteRenderer>::type)
    {
        swap = mAllocators.mSpriteRendererAllocator.destruct(index);
    
        mIdState.mSpriteRendererIdToGlobalIndex.erase(componentId);
        mIdState.mIdToGlobalIndex.erase(componentId);
        mIdState.mIdToType.erase(componentId);
    
        if (swap != nullptr)
        {
            mIdState.mSpriteRendererIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = componentType;
        }
    }
    else if (componentType == ComponentType<SpriteRenderer>::type)
    {
        swap = mAllocators.mSpriteRendererAllocator.destruct(index);
    
        mIdState.mSpriteRendererIdToGlobalIndex.erase(componentId);
        mIdState.mIdToGlobalIndex.erase(componentId);
        mIdState.mIdToType.erase(componentId);
    
        if (swap != nullptr)
        {
            mIdState.mSpriteRendererIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = componentType;
        }
    }
    else if (componentType == ComponentType<LineRenderer>::type)
    {
        swap = mAllocators.mLineRendererAllocator.destruct(index);
    
        mIdState.mLineRendererIdToGlobalIndex.erase(componentId);
        mIdState.mIdToGlobalIndex.erase(componentId);
        mIdState.mIdToType.erase(componentId);
    
        if (swap != nullptr)
        {
            mIdState.mLineRendererIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = componentType;
        }
    }
    else if (componentType == ComponentType<Light>::type)
    {
        swap = mAllocators.mLightAllocator.destruct(index);
    
        mIdState.mLightIdToGlobalIndex.erase(componentId);
        mIdState.mIdToGlobalIndex.erase(componentId);
        mIdState.mIdToType.erase(componentId);
    
        if (swap != nullptr)
        {
            mIdState.mLightIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = componentType;
        }
    }
    else if (componentType == ComponentType<BoxCollider>::type)
    {
        swap = mAllocators.mBoxColliderAllocator.destruct(index);
    
        mIdState.mBoxColliderIdToGlobalIndex.erase(componentId);
        mIdState.mIdToGlobalIndex.erase(componentId);
        mIdState.mIdToType.erase(componentId);
    
        if (swap != nullptr)
        {
            mIdState.mBoxColliderIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = componentType;
        }
    }
    else if (componentType == ComponentType<SphereCollider>::type)
    {
        swap = mAllocators.mSphereColliderAllocator.destruct(index);
    
        mIdState.mSphereColliderIdToGlobalIndex.erase(componentId);
        mIdState.mIdToGlobalIndex.erase(componentId);
        mIdState.mIdToType.erase(componentId);
    
        if (swap != nullptr)
        {
            mIdState.mSphereColliderIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = componentType;
        }
    }
    else if (componentType == ComponentType<MeshCollider>::type)
    {
        swap = mAllocators.mMeshColliderAllocator.destruct(index);
    
        mIdState.mMeshColliderIdToGlobalIndex.erase(componentId);
        mIdState.mIdToGlobalIndex.erase(componentId);
        mIdState.mIdToType.erase(componentId);
    
        if (swap != nullptr)
        {
            mIdState.mMeshColliderIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = componentType;
        }
    }
    else if (componentType == ComponentType<CapsuleCollider>::type)
    {
        swap = mAllocators.mCapsuleColliderAllocator.destruct(index);
    
        mIdState.mCapsuleColliderIdToGlobalIndex.erase(componentId);
        mIdState.mIdToGlobalIndex.erase(componentId);
        mIdState.mIdToType.erase(componentId);
    
        if (swap != nullptr)
        {
            mIdState.mCapsuleColliderIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = componentType;
        }
    }
    else if (componentType == ComponentType<Terrain>::type)
    {
        swap = mAllocators.mTerrainAllocator.destruct(index);
    
        mIdState.mTerrainIdToGlobalIndex.erase(componentId);
        mIdState.mIdToGlobalIndex.erase(componentId);
        mIdState.mIdToType.erase(componentId);
    
        if (swap != nullptr)
        {
            mIdState.mTerrainIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToGlobalIndex[swap->getId()] = index;
            mIdState.mIdToType[swap->getId()] = componentType;
        }
    }
    else
    {
        std::string message = "Error: Invalid component instance type (" + std::to_string(componentType) +
                                ") when trying to destroy internal component\n";
        Log::error(message.c_str());
    }
}

bool Scene::isMarkedForLatentDestroy(const Guid &id)
{
    for (size_t i = 0; i < mIdState.mEntityIdsMarkedLatentDestroy.size(); i++)
    {
        if (mIdState.mEntityIdsMarkedLatentDestroy[i] == id)
        {
            return true;
        }
    }

    for (size_t i = 0; i < mIdState.mComponentIdsMarkedLatentDestroy.size(); i++)
    {
        if (std::get<1>(mIdState.mComponentIdsMarkedLatentDestroy[i]) == id)
        {
            return true;
        }
    }

    return false;
}

void Scene::clearIdsMarkedCreatedOrDestroyed()
{
    mIdState.mEntityIdsMarkedCreated.clear();
    mIdState.mEntityIdsMarkedLatentDestroy.clear();
    mIdState.mComponentIdsMarkedCreated.clear();
    mIdState.mComponentIdsMarkedLatentDestroy.clear();
}

std::vector<std::pair<Guid, int>> Scene::getComponentsOnEntity(const Guid &entityId) const
{
    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::const_iterator it =
        mIdState.mEntityIdToComponentIds.find(entityId);
    if (it != mIdState.mEntityIdToComponentIds.end())
    {
        return it->second;
    }

    return std::vector<std::pair<Guid, int>>();
}

std::vector<Guid> Scene::getEntityIdsMarkedCreated() const
{
    return mIdState.mEntityIdsMarkedCreated;
}

std::vector<Guid> Scene::getEntityIdsMarkedLatentDestroy() const
{
    return mIdState.mEntityIdsMarkedLatentDestroy;
}

std::vector<std::tuple<Guid, Guid, int>> Scene::getComponentIdsMarkedCreated() const
{
    return mIdState.mComponentIdsMarkedCreated;
}

std::vector<std::tuple<Guid, Guid, int>> Scene::getComponentIdsMarkedLatentDestroy() const
{
    return mIdState.mComponentIdsMarkedLatentDestroy;
}