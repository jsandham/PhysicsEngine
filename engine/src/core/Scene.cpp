#include <fstream>

#include "../../include/core/Entity.h"
#include "../../include/core/GLM.h"
#include "../../include/core/Log.h"
#include "../../include/core/Scene.h"
#include "../../include/core/SerializationEnums.h"
#include "../../include/core/SerializationYaml.h"
#include "../../include/core/Types.h"
#include "../../include/core/Version.h"
#include "../../include/core/World.h"

#include "../../include/components/ComponentTypes.h"

using namespace PhysicsEngine;

template <typename T> static void deserializeOrLoadComponent(Scene *scene, const Guid &guid, const YAML::Node &in)
{
    static_assert(IsComponent<T>::value);

    T *component = scene->getComponentByGuid<T>(guid);
    if (component == nullptr)
    {
        component = scene->addComponent<T>(in);
    }

    if (component != nullptr)
    {
        component->deserialize(in);
    }
    else
    {
        Log::warn("A scene component could not be loaded from scene file. Skipping it.\n");
    }
}

template <typename T> static void serializeComponent(const Scene *scene, const Guid &guid, YAML::Node &out)
{
    static_assert(IsComponent<T>::value);

    T *component = scene->getComponentByGuid<T>(guid);

    if (component->mHide == HideFlag::None)
    {
        YAML::Node cn;
        component->serialize(cn);

        out[component->getObjectName()] = cn;
    }
}

template <> size_t Scene::getNumberOfComponents<Transform>() const
{
    return mAllocators.mTransformAllocator.getCount();
}

template <> size_t Scene::getNumberOfComponents<MeshRenderer>() const
{
    return mAllocators.mMeshRendererAllocator.getCount();
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

template <> Transform *Scene::getComponentByIndex<Transform>(size_t index) const
{
    return mAllocators.mTransformAllocator.get(index);
}

template <> MeshRenderer *Scene::getComponentByIndex<MeshRenderer>(size_t index) const
{
    return mAllocators.mMeshRendererAllocator.get(index);
}

template <> LineRenderer *Scene::getComponentByIndex<LineRenderer>(size_t index) const
{
    return mAllocators.mLineRendererAllocator.get(index);
}

template <> Rigidbody *Scene::getComponentByIndex<Rigidbody>(size_t index) const
{
    return mAllocators.mRigidbodyAllocator.get(index);
}

template <> Camera *Scene::getComponentByIndex<Camera>(size_t index) const
{
    return mAllocators.mCameraAllocator.get(index);
}

template <> Light *Scene::getComponentByIndex<Light>(size_t index) const
{
    return mAllocators.mLightAllocator.get(index);
}

template <> SphereCollider *Scene::getComponentByIndex<SphereCollider>(size_t index) const
{
    return mAllocators.mSphereColliderAllocator.get(index);
}

template <> BoxCollider *Scene::getComponentByIndex<BoxCollider>(size_t index) const
{
    return mAllocators.mBoxColliderAllocator.get(index);
}

template <> CapsuleCollider *Scene::getComponentByIndex<CapsuleCollider>(size_t index) const
{
    return mAllocators.mCapsuleColliderAllocator.get(index);
}

template <> MeshCollider *Scene::getComponentByIndex<MeshCollider>(size_t index) const
{
    return mAllocators.mMeshColliderAllocator.get(index);
}

template <> Terrain *Scene::getComponentByIndex<Terrain>(size_t index) const
{
    return mAllocators.mTerrainAllocator.get(index);
}

template <> Transform *Scene::getComponentById<Transform>(const Id &componentId) const
{
    return getComponentById_impl(mIdState.mTransformIdToGlobalIndex, &mAllocators.mTransformAllocator, componentId);
}

template <> MeshRenderer *Scene::getComponentById<MeshRenderer>(const Id &componentId) const
{
    return getComponentById_impl(mIdState.mMeshRendererIdToGlobalIndex, &mAllocators.mMeshRendererAllocator,
                                 componentId);
}

template <> LineRenderer *Scene::getComponentById<LineRenderer>(const Id &componentId) const
{
    return getComponentById_impl(mIdState.mLineRendererIdToGlobalIndex, &mAllocators.mLineRendererAllocator,
                                 componentId);
}

template <> Rigidbody *Scene::getComponentById<Rigidbody>(const Id &componentId) const
{
    return getComponentById_impl(mIdState.mRigidbodyIdToGlobalIndex, &mAllocators.mRigidbodyAllocator, componentId);
}

template <> Camera *Scene::getComponentById<Camera>(const Id &componentId) const
{
    return getComponentById_impl(mIdState.mCameraIdToGlobalIndex, &mAllocators.mCameraAllocator, componentId);
}

template <> Light *Scene::getComponentById<Light>(const Id &componentId) const
{
    return getComponentById_impl(mIdState.mLightIdToGlobalIndex, &mAllocators.mLightAllocator, componentId);
}

template <> SphereCollider *Scene::getComponentById<SphereCollider>(const Id &componentId) const
{
    return getComponentById_impl(mIdState.mSphereColliderIdToGlobalIndex, &mAllocators.mSphereColliderAllocator,
                                 componentId);
}

template <> BoxCollider *Scene::getComponentById<BoxCollider>(const Id &componentId) const
{
    return getComponentById_impl(mIdState.mBoxColliderIdToGlobalIndex, &mAllocators.mBoxColliderAllocator, componentId);
}

template <> CapsuleCollider *Scene::getComponentById<CapsuleCollider>(const Id &componentId) const
{
    return getComponentById_impl(mIdState.mCapsuleColliderIdToGlobalIndex, &mAllocators.mCapsuleColliderAllocator,
                                 componentId);
}

template <> MeshCollider *Scene::getComponentById<MeshCollider>(const Id &componentId) const
{
    return getComponentById_impl(mIdState.mMeshColliderIdToGlobalIndex, &mAllocators.mMeshColliderAllocator,
                                 componentId);
}

template <> Terrain *Scene::getComponentById<Terrain>(const Id &componentId) const
{
    return getComponentById_impl(mIdState.mTerrainIdToGlobalIndex, &mAllocators.mTerrainAllocator, componentId);
}

template <> Transform *Scene::getComponentByGuid<Transform>(const Guid &componentGuid) const
{
    return getComponentByGuid_impl(mIdState.mTransformGuidToGlobalIndex, &mAllocators.mTransformAllocator,
                                   componentGuid);
}

template <> MeshRenderer *Scene::getComponentByGuid<MeshRenderer>(const Guid &componentGuid) const
{
    return getComponentByGuid_impl(mIdState.mMeshRendererGuidToGlobalIndex, &mAllocators.mMeshRendererAllocator,
                                   componentGuid);
}

template <> LineRenderer *Scene::getComponentByGuid<LineRenderer>(const Guid &componentGuid) const
{
    return getComponentByGuid_impl(mIdState.mLineRendererGuidToGlobalIndex, &mAllocators.mLineRendererAllocator,
                                   componentGuid);
}

template <> Rigidbody *Scene::getComponentByGuid<Rigidbody>(const Guid &componentGuid) const
{
    return getComponentByGuid_impl(mIdState.mRigidbodyGuidToGlobalIndex, &mAllocators.mRigidbodyAllocator,
                                   componentGuid);
}

template <> Camera *Scene::getComponentByGuid<Camera>(const Guid &componentGuid) const
{
    return getComponentByGuid_impl(mIdState.mCameraGuidToGlobalIndex, &mAllocators.mCameraAllocator, componentGuid);
}

template <> Light *Scene::getComponentByGuid<Light>(const Guid &componentGuid) const
{
    return getComponentByGuid_impl(mIdState.mLightGuidToGlobalIndex, &mAllocators.mLightAllocator, componentGuid);
}

template <> SphereCollider *Scene::getComponentByGuid<SphereCollider>(const Guid &componentGuid) const
{
    return getComponentByGuid_impl(mIdState.mSphereColliderGuidToGlobalIndex, &mAllocators.mSphereColliderAllocator,
                                   componentGuid);
}

template <> BoxCollider *Scene::getComponentByGuid<BoxCollider>(const Guid &componentGuid) const
{
    return getComponentByGuid_impl(mIdState.mBoxColliderGuidToGlobalIndex, &mAllocators.mBoxColliderAllocator,
                                   componentGuid);
}

template <> CapsuleCollider *Scene::getComponentByGuid<CapsuleCollider>(const Guid &componentGuid) const
{
    return getComponentByGuid_impl(mIdState.mCapsuleColliderGuidToGlobalIndex, &mAllocators.mCapsuleColliderAllocator,
                                   componentGuid);
}

template <> MeshCollider *Scene::getComponentByGuid<MeshCollider>(const Guid &componentGuid) const
{
    return getComponentByGuid_impl(mIdState.mMeshColliderGuidToGlobalIndex, &mAllocators.mMeshColliderAllocator,
                                   componentGuid);
}

template <> Terrain *Scene::getComponentByGuid<Terrain>(const Guid &componentGuid) const
{
    return getComponentByGuid_impl(mIdState.mTerrainGuidToGlobalIndex, &mAllocators.mTerrainAllocator, componentGuid);
}

template <> Transform *Scene::getComponent<Transform>(const Guid &entityGuid) const
{
    // Transform occurs at same index as its entity since all entities have a transform
    return getComponentByIndex<Transform>(getIndexOf(entityGuid));
}

template <> MeshRenderer *Scene::getComponent<MeshRenderer>(const Guid &entityGuid) const
{
    return getComponent_impl(&mAllocators.mMeshRendererAllocator, entityGuid);
}

template <> LineRenderer *Scene::getComponent<LineRenderer>(const Guid &entityGuid) const
{
    return getComponent_impl(&mAllocators.mLineRendererAllocator, entityGuid);
}

template <> Rigidbody *Scene::getComponent<Rigidbody>(const Guid &entityGuid) const
{
    return getComponent_impl(&mAllocators.mRigidbodyAllocator, entityGuid);
}

template <> Camera *Scene::getComponent<Camera>(const Guid &entityGuid) const
{
    return getComponent_impl(&mAllocators.mCameraAllocator, entityGuid);
}

template <> Light *Scene::getComponent<Light>(const Guid &entityGuid) const
{
    return getComponent_impl(&mAllocators.mLightAllocator, entityGuid);
}

template <> SphereCollider *Scene::getComponent<SphereCollider>(const Guid &entityGuid) const
{
    return getComponent_impl(&mAllocators.mSphereColliderAllocator, entityGuid);
}

template <> BoxCollider *Scene::getComponent<BoxCollider>(const Guid &entityGuid) const
{
    return getComponent_impl(&mAllocators.mBoxColliderAllocator, entityGuid);
}

template <> CapsuleCollider *Scene::getComponent<CapsuleCollider>(const Guid &entityGuid) const
{
    return getComponent_impl(&mAllocators.mCapsuleColliderAllocator, entityGuid);
}

template <> MeshCollider *Scene::getComponent<MeshCollider>(const Guid &entityGuid) const
{
    return getComponent_impl(&mAllocators.mMeshColliderAllocator, entityGuid);
}

template <> Terrain *Scene::getComponent<Terrain>(const Guid &entityGuid) const
{
    return getComponent_impl(&mAllocators.mTerrainAllocator, entityGuid);
}

template <> MeshRenderer *Scene::addComponent<MeshRenderer>(const Guid &entityGuid)
{
    return addComponent_impl(&mAllocators.mMeshRendererAllocator, entityGuid);
}

template <> LineRenderer *Scene::addComponent<LineRenderer>(const Guid &entityGuid)
{
    return addComponent_impl(&mAllocators.mLineRendererAllocator, entityGuid);
}

template <> Rigidbody *Scene::addComponent<Rigidbody>(const Guid &entityGuid)
{
    return addComponent_impl(&mAllocators.mRigidbodyAllocator, entityGuid);
}

template <> Camera *Scene::addComponent<Camera>(const Guid &entityGuid)
{
    return addComponent_impl(&mAllocators.mCameraAllocator, entityGuid);
}

template <> Light *Scene::addComponent<Light>(const Guid &entityGuid)
{
    return addComponent_impl(&mAllocators.mLightAllocator, entityGuid);
}

template <> SphereCollider *Scene::addComponent<SphereCollider>(const Guid &entityGuid)
{
    return addComponent_impl(&mAllocators.mSphereColliderAllocator, entityGuid);
}

template <> BoxCollider *Scene::addComponent<BoxCollider>(const Guid &entityGuid)
{
    return addComponent_impl(&mAllocators.mBoxColliderAllocator, entityGuid);
}

template <> CapsuleCollider *Scene::addComponent<CapsuleCollider>(const Guid &entityGuid)
{
    return addComponent_impl(&mAllocators.mCapsuleColliderAllocator, entityGuid);
}

template <> MeshCollider *Scene::addComponent<MeshCollider>(const Guid &entityGuid)
{
    return addComponent_impl(&mAllocators.mMeshColliderAllocator, entityGuid);
}

template <> Terrain *Scene::addComponent<Terrain>(const Guid &entityGuid)
{
    return addComponent_impl(&mAllocators.mTerrainAllocator, entityGuid);
}

template <> Transform *Scene::addComponent<Transform>(const YAML::Node &in)
{
    return addComponent_impl(&mAllocators.mTransformAllocator, in);
}

template <> MeshRenderer *Scene::addComponent<MeshRenderer>(const YAML::Node &in)
{
    return addComponent_impl(&mAllocators.mMeshRendererAllocator, in);
}

template <> LineRenderer *Scene::addComponent<LineRenderer>(const YAML::Node &in)
{
    return addComponent_impl(&mAllocators.mLineRendererAllocator, in);
}

template <> Rigidbody *Scene::addComponent<Rigidbody>(const YAML::Node &in)
{
    return addComponent_impl(&mAllocators.mRigidbodyAllocator, in);
}

template <> Camera *Scene::addComponent<Camera>(const YAML::Node &in)
{
    return addComponent_impl(&mAllocators.mCameraAllocator, in);
}

template <> Light *Scene::addComponent<Light>(const YAML::Node &in)
{
    return addComponent_impl(&mAllocators.mLightAllocator, in);
}

template <> SphereCollider *Scene::addComponent<SphereCollider>(const YAML::Node &in)
{
    return addComponent_impl(&mAllocators.mSphereColliderAllocator, in);
}

template <> BoxCollider *Scene::addComponent<BoxCollider>(const YAML::Node &in)
{
    return addComponent_impl(&mAllocators.mBoxColliderAllocator, in);
}

template <> CapsuleCollider *Scene::addComponent<CapsuleCollider>(const YAML::Node &in)
{
    return addComponent_impl(&mAllocators.mCapsuleColliderAllocator, in);
}

template <> MeshCollider *Scene::addComponent<MeshCollider>(const YAML::Node &in)
{
    return addComponent_impl(&mAllocators.mMeshColliderAllocator, in);
}

template <> Terrain *Scene::addComponent<Terrain>(const YAML::Node &in)
{
    return addComponent_impl(&mAllocators.mTerrainAllocator, in);
}

void Scene::addToIdState(const Guid &guid, const Id &id, int index, int type)
{
    assert(guid != Guid::INVALID);
    assert(id != Id::INVALID);
    assert(index >= 0);

    mIdState.mGuidToGlobalIndex[guid] = index;
    mIdState.mIdToGlobalIndex[id] = index;

    mIdState.mGuidToType[guid] = type;
    mIdState.mIdToType[id] = type;

    mIdState.mGuidToId[guid] = id;
    mIdState.mIdToGuid[id] = guid;
}

void Scene::removeFromIdState(const Guid &guid, const Id &id)
{
    assert(guid != Guid::INVALID);
    assert(id != Id::INVALID);

    mIdState.mGuidToGlobalIndex.erase(guid);
    mIdState.mIdToGlobalIndex.erase(id);

    mIdState.mGuidToType.erase(guid);
    mIdState.mIdToType.erase(id);

    mIdState.mGuidToId.erase(guid);
    mIdState.mIdToGuid.erase(id);
}

template <> void Scene::addToIdState_impl<Entity>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mEntityGuidToGlobalIndex[guid] = index;
    mIdState.mEntityIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void Scene::addToIdState_impl<Transform>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mTransformGuidToGlobalIndex[guid] = index;
    mIdState.mTransformIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void Scene::addToIdState_impl<MeshRenderer>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mMeshRendererGuidToGlobalIndex[guid] = index;
    mIdState.mMeshRendererIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void Scene::addToIdState_impl<LineRenderer>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mLineRendererGuidToGlobalIndex[guid] = index;
    mIdState.mLineRendererIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void Scene::addToIdState_impl<Rigidbody>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mRigidbodyGuidToGlobalIndex[guid] = index;
    mIdState.mRigidbodyIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void Scene::addToIdState_impl<Camera>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mCameraGuidToGlobalIndex[guid] = index;
    mIdState.mCameraIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void Scene::addToIdState_impl<Light>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mLightGuidToGlobalIndex[guid] = index;
    mIdState.mLightIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void Scene::addToIdState_impl<SphereCollider>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mSphereColliderGuidToGlobalIndex[guid] = index;
    mIdState.mSphereColliderIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void Scene::addToIdState_impl<BoxCollider>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mBoxColliderGuidToGlobalIndex[guid] = index;
    mIdState.mBoxColliderIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void Scene::addToIdState_impl<CapsuleCollider>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mCapsuleColliderGuidToGlobalIndex[guid] = index;
    mIdState.mCapsuleColliderIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void Scene::addToIdState_impl<MeshCollider>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mMeshColliderGuidToGlobalIndex[guid] = index;
    mIdState.mMeshColliderIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void Scene::addToIdState_impl<Terrain>(const Guid &guid, const Id &id, int index, int type)
{
    mIdState.mTerrainGuidToGlobalIndex[guid] = index;
    mIdState.mTerrainIdToGlobalIndex[id] = index;

    addToIdState(guid, id, index, type);
}

template <> void Scene::removeFromIdState_impl<Entity>(const Guid &guid, const Id &id)
{
    mIdState.mEntityGuidToGlobalIndex.erase(guid);
    mIdState.mEntityIdToGlobalIndex.erase(id);

    removeFromIdState(guid, id);
}

template <> void Scene::removeFromIdState_impl<Transform>(const Guid &guid, const Id &id)
{
    mIdState.mTransformGuidToGlobalIndex.erase(guid);
    mIdState.mTransformIdToGlobalIndex.erase(id);

    removeFromIdState(guid, id);
}

template <> void Scene::removeFromIdState_impl<MeshRenderer>(const Guid &guid, const Id &id)
{
    mIdState.mMeshRendererGuidToGlobalIndex.erase(guid);
    mIdState.mMeshRendererIdToGlobalIndex.erase(id);

    removeFromIdState(guid, id);
}

template <> void Scene::removeFromIdState_impl<LineRenderer>(const Guid &guid, const Id &id)
{
    mIdState.mLineRendererGuidToGlobalIndex.erase(guid);
    mIdState.mLineRendererIdToGlobalIndex.erase(id);

    removeFromIdState(guid, id);
}

template <> void Scene::removeFromIdState_impl<Rigidbody>(const Guid &guid, const Id &id)
{
    mIdState.mRigidbodyGuidToGlobalIndex.erase(guid);
    mIdState.mRigidbodyIdToGlobalIndex.erase(id);

    removeFromIdState(guid, id);
}

template <> void Scene::removeFromIdState_impl<Camera>(const Guid &guid, const Id &id)
{
    mIdState.mCameraGuidToGlobalIndex.erase(guid);
    mIdState.mCameraIdToGlobalIndex.erase(id);

    removeFromIdState(guid, id);
}

template <> void Scene::removeFromIdState_impl<Light>(const Guid &guid, const Id &id)
{
    mIdState.mLightGuidToGlobalIndex.erase(guid);
    mIdState.mLightIdToGlobalIndex.erase(id);

    removeFromIdState(guid, id);
}

template <> void Scene::removeFromIdState_impl<SphereCollider>(const Guid &guid, const Id &id)
{
    mIdState.mSphereColliderGuidToGlobalIndex.erase(guid);
    mIdState.mSphereColliderIdToGlobalIndex.erase(id);

    removeFromIdState(guid, id);
}

template <> void Scene::removeFromIdState_impl<BoxCollider>(const Guid &guid, const Id &id)
{
    mIdState.mBoxColliderGuidToGlobalIndex.erase(guid);
    mIdState.mBoxColliderIdToGlobalIndex.erase(id);

    removeFromIdState(guid, id);
}

template <> void Scene::removeFromIdState_impl<CapsuleCollider>(const Guid &guid, const Id &id)
{
    mIdState.mCapsuleColliderGuidToGlobalIndex.erase(guid);
    mIdState.mCapsuleColliderIdToGlobalIndex.erase(id);

    removeFromIdState(guid, id);
}

template <> void Scene::removeFromIdState_impl<MeshCollider>(const Guid &guid, const Id &id)
{
    mIdState.mMeshColliderGuidToGlobalIndex.erase(guid);
    mIdState.mMeshColliderIdToGlobalIndex.erase(id);

    removeFromIdState(guid, id);
}

template <> void Scene::removeFromIdState_impl<Terrain>(const Guid &guid, const Id &id)
{
    mIdState.mTerrainGuidToGlobalIndex.erase(guid);
    mIdState.mTerrainIdToGlobalIndex.erase(id);

    removeFromIdState(guid, id);
}

template <typename T>
T *Scene::getComponentById_impl(const std::unordered_map<Id, int> &idToIndexMap, const PoolAllocator<T> *allocator,
                                const Id &id) const
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

template <typename T>
T *Scene::getComponentByGuid_impl(const std::unordered_map<Guid, int> &guidToIndexMap,
                                  const PoolAllocator<T> *allocator, const Guid &guid) const
{
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

template <typename T> T *Scene::getComponent_impl(const PoolAllocator<T> *allocator, const Guid &entityGuid) const
{
    static_assert(IsComponent<T>::value, "'T' is not of type Component");

    assert(allocator != nullptr);

    std::vector<std::pair<Guid, int>> componentsOnEntity = getComponentsOnEntity(entityGuid);

    for (size_t i = 0; i < componentsOnEntity.size(); i++)
    {
        if (ComponentType<T>::type == componentsOnEntity[i].second)
        {
            std::unordered_map<Guid, int>::const_iterator it =
                mIdState.mGuidToGlobalIndex.find(componentsOnEntity[i].first);
            if (it != mIdState.mGuidToGlobalIndex.end())
            {
                return allocator->get(it->second);
            }

            break;
        }
    }

    return nullptr;
}

template <typename T> T *Scene::addComponent_impl(PoolAllocator<T> *allocator, const YAML::Node &in)
{
    static_assert(IsComponent<T>::value, "'T' is not of type Component");

    assert(allocator != nullptr);

    int componentGlobalIndex = (int)allocator->getCount();
    int componentType = ComponentType<T>::type;

    T *component = allocator->construct(mWorld, in, Id::newId());

    if (component != nullptr)
    {
        addToIdState_impl<T>(component->getGuid(), component->getId(), componentGlobalIndex, componentType);

        mIdState.mEntityGuidToComponentIds[component->getEntityGuid()].push_back(
            std::make_pair(component->getGuid(), componentType));

        mIdState.mComponentGuidsMarkedCreated.push_back(
            std::make_tuple(component->getEntityGuid(), component->getGuid(), componentType));
    }

    return component;
}

template <typename T> T *Scene::addComponent_impl(PoolAllocator<T> *allocator, const Guid &entityGuid)
{
    static_assert(IsComponent<T>::value, "'T' is not of type Component");

    assert(allocator != nullptr);

    if (getTypeOf(entityGuid) != EntityType<Entity>::type)
    {
        return nullptr;
    }

    int componentGlobalIndex = (int)allocator->getCount();
    int componentType = ComponentType<T>::type;

    T *component = allocator->construct(mWorld, Guid::newGuid(), Id::newId());

    if (component != nullptr)
    {
        component->mEntityGuid = entityGuid;

        addToIdState_impl<T>(component->getGuid(), component->getId(), componentGlobalIndex, componentType);

        mIdState.mEntityGuidToComponentIds[entityGuid].push_back(std::make_pair(component->getGuid(), componentType));

        mIdState.mComponentGuidsMarkedCreated.push_back(
            std::make_tuple(entityGuid, component->getGuid(), componentType));
    }

    return component;
}

Scene::Scene(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mName = "Unnamed scene";
    mVersion = SCENE_VERSION;
}

Scene::Scene(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mName = "Unnamed scene";
    mVersion = SCENE_VERSION;
}

Scene::~Scene()
{
}

void Scene::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["name"] = mName;
    out["version"] = mVersion;
}

void Scene::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

    mName = YAML::getValue<std::string>(in, "name");
    mVersion = YAML::getValue<std::string>(in, "version");

    assert(getGuid() == YAML::getValue<Guid>(in, "id"));

    // load all objects found in scene file
    for (YAML::const_iterator it = in.begin(); it != in.end(); ++it)
    {
        if (it->first.IsScalar() && it->second.IsMap())
        {
            int objectType = YAML::getValue<int>(it->second, "type");
            Guid objectGuid = YAML::getValue<Guid>(it->second, "id");
            HideFlag objectHideFlag = YAML::getValue<HideFlag>(it->second, "hide");

            if (objectHideFlag != HideFlag::DontSave)
            {
                if (isEntity(objectType))
                {
                    Entity *entity = getEntityByGuid(objectGuid);
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
                    switch (objectType)
                    {
                    case ComponentType<Transform>::type: {
                        deserializeOrLoadComponent<Transform>(this, objectGuid, it->second);
                        break;
                    }
                    case ComponentType<Rigidbody>::type: {
                        deserializeOrLoadComponent<Rigidbody>(this, objectGuid, it->second);
                        break;
                    }
                    case ComponentType<Camera>::type: {
                        deserializeOrLoadComponent<Rigidbody>(this, objectGuid, it->second);
                        break;
                    }
                    case ComponentType<MeshRenderer>::type: {
                        deserializeOrLoadComponent<MeshRenderer>(this, objectGuid, it->second);
                        break;
                    }
                    case ComponentType<LineRenderer>::type: {
                        deserializeOrLoadComponent<LineRenderer>(this, objectGuid, it->second);
                        break;
                    }
                    case ComponentType<Light>::type: {
                        deserializeOrLoadComponent<Light>(this, objectGuid, it->second);
                        break;
                    }
                    case ComponentType<BoxCollider>::type: {
                        deserializeOrLoadComponent<BoxCollider>(this, objectGuid, it->second);
                        break;
                    }
                    case ComponentType<SphereCollider>::type: {
                        deserializeOrLoadComponent<SphereCollider>(this, objectGuid, it->second);
                        break;
                    }
                    case ComponentType<MeshCollider>::type: {
                        deserializeOrLoadComponent<MeshCollider>(this, objectGuid, it->second);
                        break;
                    }
                    case ComponentType<CapsuleCollider>::type: {
                        deserializeOrLoadComponent<CapsuleCollider>(this, objectGuid, it->second);
                        break;
                    }
                    case ComponentType<Terrain>::type: {
                        deserializeOrLoadComponent<Terrain>(this, objectGuid, it->second);
                        break;
                    }
                    default:
                        assert(!"Unreachable code");
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

Guid Scene::getGuid() const
{
    return mGuid;
}

Id Scene::getId() const
{
    return mId;
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
                YAML::Node componentNode;
                switch (temp[j].second)
                {
                case ComponentType<Transform>::type: {
                    serializeComponent<Transform>(this, temp[j].first, componentNode);
                    break;
                }
                case ComponentType<Rigidbody>::type: {
                    serializeComponent<Rigidbody>(this, temp[j].first, componentNode);
                    break;
                }
                case ComponentType<Camera>::type: {
                    serializeComponent<Camera>(this, temp[j].first, componentNode);
                    break;
                }
                case ComponentType<MeshRenderer>::type: {
                    serializeComponent<MeshRenderer>(this, temp[j].first, componentNode);
                    break;
                }
                case ComponentType<LineRenderer>::type: {
                    serializeComponent<LineRenderer>(this, temp[j].first, componentNode);
                    break;
                }
                case ComponentType<Light>::type: {
                    serializeComponent<Light>(this, temp[j].first, componentNode);
                    break;
                }
                case ComponentType<BoxCollider>::type: {
                    serializeComponent<BoxCollider>(this, temp[j].first, componentNode);
                    break;
                }
                case ComponentType<SphereCollider>::type: {
                    serializeComponent<SphereCollider>(this, temp[j].first, componentNode);
                    break;
                }
                case ComponentType<MeshCollider>::type: {
                    serializeComponent<MeshCollider>(this, temp[j].first, componentNode);
                    break;
                }
                case ComponentType<CapsuleCollider>::type: {
                    serializeComponent<CapsuleCollider>(this, temp[j].first, componentNode);
                    break;
                }
                case ComponentType<Terrain>::type: {
                    serializeComponent<Terrain>(this, temp[j].first, componentNode);
                    break;
                }
                default:
                    assert(!"Unreachable code");
                }

                out << componentNode;
                out << "\n";
            }
        }
    }

    out.close();

    return true;
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

Entity *Scene::getEntityById(const Id &entityId) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mEntityIdToGlobalIndex.find(entityId);
    if (it != mIdState.mEntityIdToGlobalIndex.end())
    {
        return mAllocators.mEntityAllocator.get(it->second);
    }
    else
    {
        return nullptr;
    }
}

Entity *Scene::getEntityByGuid(const Guid &entityGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mEntityGuidToGlobalIndex.find(entityGuid);
    if (it != mIdState.mEntityGuidToGlobalIndex.end())
    {
        return mAllocators.mEntityAllocator.get(it->second);
    }
    else
    {
        return nullptr;
    }
}

int Scene::getIndexOf(const Id &id) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mIdToGlobalIndex.find(id);
    if (it != mIdState.mIdToGlobalIndex.end())
    {
        return it->second;
    }

    return -1;
}

int Scene::getTypeOf(const Id &id) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mIdToType.find(id);
    if (it != mIdState.mIdToType.end())
    {
        return it->second;
    }

    return -1;
}

int Scene::getIndexOf(const Guid &guid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mGuidToGlobalIndex.find(guid);
    if (it != mIdState.mGuidToGlobalIndex.end())
    {
        return it->second;
    }

    return -1;
}

int Scene::getTypeOf(const Guid &guid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mGuidToType.find(guid);
    if (it != mIdState.mGuidToType.end())
    {
        return it->second;
    }

    return -1;
}

Guid Scene::getGuidFromId(const Id &id) const
{
    std::unordered_map<Id, Guid>::const_iterator it = mIdState.mIdToGuid.find(id);
    if (it != mIdState.mIdToGuid.end())
    {
        return it->second;
    }

    return Guid::INVALID;
}

Id Scene::getIdFromGuid(const Guid &guid) const
{
    std::unordered_map<Guid, Id>::const_iterator it = mIdState.mGuidToId.find(guid);
    if (it != mIdState.mGuidToId.end())
    {
        return it->second;
    }

    return Id::INVALID;
}

void Scene::latentDestroyEntitiesInScene()
{
    // latent destroy all entities (and thereby also all components)
    for (size_t i = 0; i < getNumberOfEntities(); i++)
    {
        Entity *entity = getEntityByIndex(i);

        if (!entity->mDoNotDestroy)
        {
            latentDestroyEntity(entity->getGuid());
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
            entitiesToDestroy.push_back(entity->getGuid());
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

    Entity *entity = mAllocators.mEntityAllocator.construct(mWorld, Guid::newGuid(), Id::newId());

    if (entity != nullptr)
    {
        addToIdState_impl<Entity>(entity->getGuid(), entity->getId(), globalIndex, type);

        mIdState.mEntityGuidToComponentIds[entity->getGuid()] = std::vector<std::pair<Guid, int>>();

        mIdState.mEntityGuidsMarkedCreated.push_back(entity->getGuid());
    }

    // Add transform (all entities must have a transform)
    int componentGlobalIndex = (int)mAllocators.mTransformAllocator.getCount();
    int componentType = ComponentType<Transform>::type;

    Transform *component = mAllocators.mTransformAllocator.construct(mWorld, Guid::newGuid(), Id::newId());

    assert(component != nullptr);

    component->mEntityGuid = entity->getGuid();

    addToIdState_impl<Transform>(component->getGuid(), component->getId(), componentGlobalIndex, componentType);

    mIdState.mEntityGuidToComponentIds[entity->getGuid()].push_back(
        std::make_pair(component->getGuid(), componentType));

    mIdState.mComponentGuidsMarkedCreated.push_back(
        std::make_tuple(entity->getGuid(), component->getGuid(), componentType));

    return entity;
}

Entity *Scene::createEntity(
    const YAML::Node &in) // currently create entity without transform...pass in transform node as well??
{
    int globalIndex = (int)mAllocators.mEntityAllocator.getCount();
    int type = EntityType<Entity>::type;

    Entity *entity = mAllocators.mEntityAllocator.construct(mWorld, in, Id::newId());

    if (entity != nullptr)
    {
        addToIdState_impl<Entity>(entity->getGuid(), entity->getId(), globalIndex, type);

        mIdState.mEntityGuidToComponentIds[entity->getGuid()] = std::vector<std::pair<Guid, int>>();

        mIdState.mEntityGuidsMarkedCreated.push_back(entity->getGuid());
    }

    //// Add transform (all entities must have a transform)
    // int componentGlobalIndex = (int)mAllocators.mTransformAllocator.getCount();
    // int componentType = ComponentType<Transform>::type;

    // Transform *component = mAllocators.mTransformAllocator.construct(mWorld, Guid::newGuid());

    // assert(component != nullptr);

    // component->mEntityId = entity->getGuid();

    // addToIdState_impl<Transform>(component->getGuid(), component->getId(), componentGlobalIndex, componentType);

    // mIdState.mEntityIdToComponentIds[entity->getGuid()].push_back(std::make_pair(component->getGuid(),
    // componentType));

    // mIdState.mComponentIdsMarkedCreated.push_back(std::make_tuple(entity->getGuid(), component->getGuid(),
    // componentType));

    return entity;
}

Entity *Scene::createEntity(const std::string &name)
{
    Entity *entity = createEntity();
    if (entity != nullptr)
    {
        entity->mName = name;
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

    entity->mName = mesh->mName;

    transform->setPosition(glm::vec3(0, 0, 0));
    transform->setRotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    transform->setScale(glm::vec3(1, 1, 1));
    meshRenderer->setMesh(mesh->getGuid());
    meshRenderer->setMaterial(mWorld->getPrimtiveMaterial()->getGuid());

    return entity;
}

Entity *Scene::createNonPrimitive(const Guid &meshGuid)
{
    Mesh *mesh = mWorld->getAssetByGuid<Mesh>(meshGuid);
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

    entity->mName = mesh->mName;

    transform->setPosition(glm::vec3(0, 0, 0));
    transform->setRotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    transform->setScale(glm::vec3(1, 1, 1));
    meshRenderer->setMesh(meshGuid);
    meshRenderer->setMaterial(mWorld->getPrimtiveMaterial()->getGuid());

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

void Scene::latentDestroyEntity(const Guid &entityGuid)
{
    mIdState.mEntityGuidsMarkedLatentDestroy.push_back(entityGuid);

    // add any components found on the entity to the latent destroy component list
    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::const_iterator it =
        mIdState.mEntityGuidToComponentIds.find(entityGuid);

    assert(it != mIdState.mEntityGuidToComponentIds.end());

    for (size_t i = 0; i < it->second.size(); i++)
    {
        latentDestroyComponent(entityGuid, it->second[i].first, it->second[i].second);
    }
}

void Scene::immediateDestroyEntity(const Guid &entityGuid)
{
    // Destroy components on entity
    std::vector<std::pair<Guid, int>> componentsOnEntity = mIdState.mEntityGuidToComponentIds[entityGuid];
    for (size_t i = 0; i < componentsOnEntity.size(); i++)
    {
        immediateDestroyComponent(entityGuid, componentsOnEntity[i].first, componentsOnEntity[i].second);
    }

    assert(mIdState.mEntityGuidToComponentIds[entityGuid].size() == 0);

    mIdState.mEntityGuidToComponentIds.erase(entityGuid);

    // Destroy entity
    int index = getIndexOf(entityGuid);
    Id entityId = getIdFromGuid(entityGuid);

    Entity *swap = mAllocators.mEntityAllocator.destruct(index);

    removeFromIdState_impl<Entity>(entityGuid, entityId);

    if (swap != nullptr)
    {
        addToIdState_impl<Entity>(swap->getGuid(), swap->getId(), index, EntityType<Entity>::type);
    }
}

void Scene::latentDestroyComponent(const Guid &entityGuid, const Guid &componentGuid, int componentType)
{
    mIdState.mComponentGuidsMarkedLatentDestroy.push_back(std::make_tuple(entityGuid, componentGuid, componentType));
}

void Scene::immediateDestroyComponent(const Guid &entityGuid, const Guid &componentGuid, int componentType)
{
    // remove from entity component list
    std::vector<std::pair<Guid, int>> &componentsOnEntity = mIdState.mEntityGuidToComponentIds[entityGuid];

    std::vector<std::pair<Guid, int>>::iterator it = componentsOnEntity.begin();
    while (it < componentsOnEntity.end())
    {
        if (it->second == componentType && it->first == componentGuid)
        {
            break;
        }

        it++;
    }

    if (it < componentsOnEntity.end())
    {
        componentsOnEntity.erase(it);
    }

    // Destroy component
    int index = getIndexOf(componentGuid);
    Id componentId = getIdFromGuid(componentGuid);

    switch (componentType)
    {
    case ComponentType<Transform>::type: {
        Transform *swap = mAllocators.mTransformAllocator.destruct(index);

        removeFromIdState_impl<Transform>(componentGuid, componentId);

        if (swap != nullptr)
        {
            addToIdState_impl<Transform>(swap->getGuid(), swap->getId(), index, componentType);
        }
        break;
    }
    case ComponentType<Rigidbody>::type: {
        Rigidbody *swap = mAllocators.mRigidbodyAllocator.destruct(index);

        removeFromIdState_impl<Rigidbody>(componentGuid, componentId);

        if (swap != nullptr)
        {
            addToIdState_impl<Rigidbody>(swap->getGuid(), swap->getId(), index, componentType);
        }
        break;
    }
    case ComponentType<Camera>::type: {
        Camera *swap = mAllocators.mCameraAllocator.destruct(index);

        removeFromIdState_impl<Camera>(componentGuid, componentId);

        if (swap != nullptr)
        {
            addToIdState_impl<Camera>(swap->getGuid(), swap->getId(), index, componentType);
        }
        break;
    }
    case ComponentType<MeshRenderer>::type: {
        MeshRenderer *swap = mAllocators.mMeshRendererAllocator.destruct(index);

        removeFromIdState_impl<MeshRenderer>(componentGuid, componentId);

        if (swap != nullptr)
        {
            addToIdState_impl<MeshRenderer>(swap->getGuid(), swap->getId(), index, componentType);
        }
        break;
    }
    case ComponentType<LineRenderer>::type: {
        LineRenderer *swap = mAllocators.mLineRendererAllocator.destruct(index);

        removeFromIdState_impl<LineRenderer>(componentGuid, componentId);

        if (swap != nullptr)
        {
            addToIdState_impl<LineRenderer>(swap->getGuid(), swap->getId(), index, componentType);
        }
        break;
    }
    case ComponentType<Light>::type: {
        Light *swap = mAllocators.mLightAllocator.destruct(index);

        removeFromIdState_impl<Light>(componentGuid, componentId);

        if (swap != nullptr)
        {
            addToIdState_impl<Light>(swap->getGuid(), swap->getId(), index, componentType);
        }
        break;
    }
    case ComponentType<BoxCollider>::type: {
        BoxCollider *swap = mAllocators.mBoxColliderAllocator.destruct(index);

        removeFromIdState_impl<BoxCollider>(componentGuid, componentId);

        if (swap != nullptr)
        {
            addToIdState_impl<BoxCollider>(swap->getGuid(), swap->getId(), index, componentType);
        }
        break;
    }
    case ComponentType<SphereCollider>::type: {
        SphereCollider *swap = mAllocators.mSphereColliderAllocator.destruct(index);

        removeFromIdState_impl<SphereCollider>(componentGuid, componentId);

        if (swap != nullptr)
        {
            addToIdState_impl<SphereCollider>(swap->getGuid(), swap->getId(), index, componentType);
        }
        break;
    }
    case ComponentType<MeshCollider>::type: {
        MeshCollider *swap = mAllocators.mMeshColliderAllocator.destruct(index);

        removeFromIdState_impl<MeshCollider>(componentGuid, componentId);

        if (swap != nullptr)
        {
            addToIdState_impl<MeshCollider>(swap->getGuid(), swap->getId(), index, componentType);
        }
        break;
    }
    case ComponentType<CapsuleCollider>::type: {
        CapsuleCollider *swap = mAllocators.mCapsuleColliderAllocator.destruct(index);

        removeFromIdState_impl<CapsuleCollider>(componentGuid, componentId);

        if (swap != nullptr)
        {
            addToIdState_impl<CapsuleCollider>(swap->getGuid(), swap->getId(), index, componentType);
        }
        break;
    }
    case ComponentType<Terrain>::type: {
        Terrain *swap = mAllocators.mTerrainAllocator.destruct(index);

        removeFromIdState_impl<Terrain>(componentGuid, componentId);

        if (swap != nullptr)
        {
            addToIdState_impl<Terrain>(swap->getGuid(), swap->getId(), index, componentType);
        }
        break;
    }
    default:
        assert(!"Unreachable code");
    }
}

bool Scene::isMarkedForLatentDestroy(const Guid &guid)
{
    for (size_t i = 0; i < mIdState.mEntityGuidsMarkedLatentDestroy.size(); i++)
    {
        if (mIdState.mEntityGuidsMarkedLatentDestroy[i] == guid)
        {
            return true;
        }
    }

    for (size_t i = 0; i < mIdState.mComponentGuidsMarkedLatentDestroy.size(); i++)
    {
        if (std::get<1>(mIdState.mComponentGuidsMarkedLatentDestroy[i]) == guid)
        {
            return true;
        }
    }

    return false;
}

void Scene::clearIdsMarkedCreatedOrDestroyed()
{
    mIdState.mEntityGuidsMarkedCreated.clear();
    mIdState.mEntityGuidsMarkedLatentDestroy.clear();
    mIdState.mComponentGuidsMarkedCreated.clear();
    mIdState.mComponentGuidsMarkedLatentDestroy.clear();
}

std::vector<std::pair<Guid, int>> Scene::getComponentsOnEntity(const Guid &entityGuid) const
{
    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::const_iterator it =
        mIdState.mEntityGuidToComponentIds.find(entityGuid);
    if (it != mIdState.mEntityGuidToComponentIds.end())
    {
        return it->second;
    }

    return std::vector<std::pair<Guid, int>>();
}

std::vector<Guid> Scene::getEntityIdsMarkedCreated() const
{
    return mIdState.mEntityGuidsMarkedCreated;
}

std::vector<Guid> Scene::getEntityIdsMarkedLatentDestroy() const
{
    return mIdState.mEntityGuidsMarkedLatentDestroy;
}

std::vector<std::tuple<Guid, Guid, int>> Scene::getComponentIdsMarkedCreated() const
{
    return mIdState.mComponentGuidsMarkedCreated;
}

std::vector<std::tuple<Guid, Guid, int>> Scene::getComponentIdsMarkedLatentDestroy() const
{
    return mIdState.mComponentGuidsMarkedLatentDestroy;
}