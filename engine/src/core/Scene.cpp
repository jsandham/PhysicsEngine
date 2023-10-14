#include <fstream>
#include <iostream>

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

template <> Terrain *Scene::getComponentByIndex<Terrain>(size_t index) const
{
    return mAllocators.mTerrainAllocator.get(index);
}

template <> Transform *Scene::getComponentById<Transform>(const Id &componentId) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mTransformIdToGlobalIndex.find(componentId);
    return (it != mIdState.mTransformIdToGlobalIndex.end()) ? mAllocators.mTransformAllocator.get(it->second) : nullptr;
}

template <> MeshRenderer *Scene::getComponentById<MeshRenderer>(const Id &componentId) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mMeshRendererIdToGlobalIndex.find(componentId);
    return (it != mIdState.mMeshRendererIdToGlobalIndex.end()) ? mAllocators.mMeshRendererAllocator.get(it->second)
                                                               : nullptr;
}

template <> Rigidbody *Scene::getComponentById<Rigidbody>(const Id &componentId) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mRigidbodyIdToGlobalIndex.find(componentId);
    return (it != mIdState.mRigidbodyIdToGlobalIndex.end()) ? mAllocators.mRigidbodyAllocator.get(it->second)
                                                               : nullptr;
}

template <> Camera *Scene::getComponentById<Camera>(const Id &componentId) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mCameraIdToGlobalIndex.find(componentId);
    return (it != mIdState.mCameraIdToGlobalIndex.end()) ? mAllocators.mCameraAllocator.get(it->second) : nullptr;
}

template <> Light *Scene::getComponentById<Light>(const Id &componentId) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mLightIdToGlobalIndex.find(componentId);
    return (it != mIdState.mLightIdToGlobalIndex.end()) ? mAllocators.mLightAllocator.get(it->second) : nullptr;
}

template <> SphereCollider *Scene::getComponentById<SphereCollider>(const Id &componentId) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mSphereColliderIdToGlobalIndex.find(componentId);
    return (it != mIdState.mSphereColliderIdToGlobalIndex.end()) ? mAllocators.mSphereColliderAllocator.get(it->second)
                                                                 : nullptr;
}

template <> BoxCollider *Scene::getComponentById<BoxCollider>(const Id &componentId) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mBoxColliderIdToGlobalIndex.find(componentId);
    return (it != mIdState.mBoxColliderIdToGlobalIndex.end()) ? mAllocators.mBoxColliderAllocator.get(it->second)
                                                                 : nullptr;
}

template <> Terrain *Scene::getComponentById<Terrain>(const Id &componentId) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mTerrainIdToGlobalIndex.find(componentId);
    return (it != mIdState.mTerrainIdToGlobalIndex.end()) ? mAllocators.mTerrainAllocator.get(it->second)
                                                               : nullptr;
}

template <> Transform *Scene::getComponentByGuid<Transform>(const Guid &componentGuid) const
{
    std::string test = componentGuid.toString();
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mTransformGuidToGlobalIndex.find(componentGuid);
    return (it != mIdState.mTransformGuidToGlobalIndex.end()) ? mAllocators.mTransformAllocator.get(it->second)
                                                              : nullptr;
}

template <> MeshRenderer *Scene::getComponentByGuid<MeshRenderer>(const Guid &componentGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mMeshRendererGuidToGlobalIndex.find(componentGuid);
    return (it != mIdState.mMeshRendererGuidToGlobalIndex.end()) ? mAllocators.mMeshRendererAllocator.get(it->second)
                                                              : nullptr;
}

template <> Rigidbody *Scene::getComponentByGuid<Rigidbody>(const Guid &componentGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mRigidbodyGuidToGlobalIndex.find(componentGuid);
    return (it != mIdState.mRigidbodyGuidToGlobalIndex.end()) ? mAllocators.mRigidbodyAllocator.get(it->second)
                                                                 : nullptr;
}

template <> Camera *Scene::getComponentByGuid<Camera>(const Guid &componentGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mCameraGuidToGlobalIndex.find(componentGuid);
    return (it != mIdState.mCameraGuidToGlobalIndex.end()) ? mAllocators.mCameraAllocator.get(it->second)
                                                              : nullptr;
}

template <> Light *Scene::getComponentByGuid<Light>(const Guid &componentGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mLightGuidToGlobalIndex.find(componentGuid);
    return (it != mIdState.mLightGuidToGlobalIndex.end()) ? mAllocators.mLightAllocator.get(it->second) : nullptr;
}

template <> SphereCollider *Scene::getComponentByGuid<SphereCollider>(const Guid &componentGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mSphereColliderGuidToGlobalIndex.find(componentGuid);
    return (it != mIdState.mSphereColliderGuidToGlobalIndex.end())
               ? mAllocators.mSphereColliderAllocator.get(it->second)
                                                                   : nullptr;
}

template <> BoxCollider *Scene::getComponentByGuid<BoxCollider>(const Guid &componentGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mBoxColliderGuidToGlobalIndex.find(componentGuid);
    return (it != mIdState.mBoxColliderGuidToGlobalIndex.end()) ? mAllocators.mBoxColliderAllocator.get(it->second)
               : nullptr;
}

template <> Terrain *Scene::getComponentByGuid<Terrain>(const Guid &componentGuid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mTerrainGuidToGlobalIndex.find(componentGuid);
    return (it != mIdState.mTerrainGuidToGlobalIndex.end()) ? mAllocators.mTerrainAllocator.get(it->second)
                                                                 : nullptr;
}

template <> Transform *Scene::getComponent<Transform>(const Guid &entityGuid) const
{
    // Transform occurs at same index as its entity since all entities have a transform
    return getComponentByIndex<Transform>(getIndexOf(entityGuid));
}

template <> MeshRenderer *Scene::getComponent<MeshRenderer>(const Guid &entityGuid) const
{
    int index = getComponentIndex(entityGuid, ComponentType<MeshRenderer>::type);
    return index >= 0 ? mAllocators.mMeshRendererAllocator.get(index) : nullptr; 
}

template <> Rigidbody *Scene::getComponent<Rigidbody>(const Guid &entityGuid) const
{
    int index = getComponentIndex(entityGuid, ComponentType<Rigidbody>::type);
    return index >= 0 ? mAllocators.mRigidbodyAllocator.get(index) : nullptr; 
}

template <> Camera *Scene::getComponent<Camera>(const Guid &entityGuid) const
{
    int index = getComponentIndex(entityGuid, ComponentType<Camera>::type);
    return index >= 0 ? mAllocators.mCameraAllocator.get(index) : nullptr; 
}

template <> Light *Scene::getComponent<Light>(const Guid &entityGuid) const
{
    int index = getComponentIndex(entityGuid, ComponentType<Light>::type);
    return index >= 0 ? mAllocators.mLightAllocator.get(index) : nullptr; 
}

template <> SphereCollider *Scene::getComponent<SphereCollider>(const Guid &entityGuid) const
{
    int index = getComponentIndex(entityGuid, ComponentType<SphereCollider>::type);
    return index >= 0 ? mAllocators.mSphereColliderAllocator.get(index) : nullptr; 
}

template <> BoxCollider *Scene::getComponent<BoxCollider>(const Guid &entityGuid) const
{
    int index = getComponentIndex(entityGuid, ComponentType<BoxCollider>::type);
    return index >= 0 ? mAllocators.mBoxColliderAllocator.get(index) : nullptr; 
}

template <> Terrain *Scene::getComponent<Terrain>(const Guid &entityGuid) const
{
    int index = getComponentIndex(entityGuid, ComponentType<Terrain>::type);
    return index >= 0 ? mAllocators.mTerrainAllocator.get(index) : nullptr; 
}

template <> MeshRenderer *Scene::addComponent<MeshRenderer>(const Guid &entityGuid)
{
    return addMeshRenderer(entityGuid);
}

template <> Rigidbody *Scene::addComponent<Rigidbody>(const Guid &entityGuid)
{
    return addRigidbody(entityGuid);
}

template <> Camera *Scene::addComponent<Camera>(const Guid &entityGuid)
{
    return addCamera(entityGuid);
}

template <> Light *Scene::addComponent<Light>(const Guid &entityGuid)
{
    return addLight(entityGuid);
}

template <> SphereCollider *Scene::addComponent<SphereCollider>(const Guid &entityGuid)
{
    return addSphereCollider(entityGuid);
}

template <> BoxCollider *Scene::addComponent<BoxCollider>(const Guid &entityGuid)
{
    return addBoxCollider(entityGuid);
}

template <> Terrain *Scene::addComponent<Terrain>(const Guid &entityGuid)
{
    return addTerrain(entityGuid);
}

template <> Transform *Scene::addComponent<Transform>(const YAML::Node &in)
{
    return addTransform(in);
}

template <> MeshRenderer *Scene::addComponent<MeshRenderer>(const YAML::Node &in)
{
    return addMeshRenderer(in);
}

template <> Rigidbody *Scene::addComponent<Rigidbody>(const YAML::Node &in)
{
    return addRigidbody(in);
}

template <> Camera *Scene::addComponent<Camera>(const YAML::Node &in)
{
    return addCamera(in);
}

template <> Light *Scene::addComponent<Light>(const YAML::Node &in)
{
    return addLight(in);
}

template <> SphereCollider *Scene::addComponent<SphereCollider>(const YAML::Node &in)
{
    return addSphereCollider(in);
}

template <> BoxCollider *Scene::addComponent<BoxCollider>(const YAML::Node &in)
{
    return addBoxCollider(in);
}

template <> Terrain *Scene::addComponent<Terrain>(const YAML::Node &in)
{
    return addTerrain(in);
}

int Scene::getComponentIndex(const Guid &entityGuid, int componentType) const
{
    std::vector<std::pair<Guid, int>> componentsOnEntity = getComponentsOnEntity(entityGuid);
    
    for (size_t i = 0; i < componentsOnEntity.size(); i++)
    {
        if (componentType == componentsOnEntity[i].second)
        {
            std::unordered_map<Guid, int>::const_iterator it =
                mIdState.mGuidToGlobalIndex.find(componentsOnEntity[i].first);
            if (it != mIdState.mGuidToGlobalIndex.end())
            {
                return it->second;
            }
    
            break;
        }
    }

    return -1;
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
    assert(mAllocators.mEntityAllocator.getCount() == mAllocators.mTransformAllocator.getCount());
    assert(mAllocators.mEntityAllocator.getCount() == mAllocators.mTransformDataAllocator.getCount());

    std::cout << "Before immediate destroy" << std::endl;
    std::cout << "mEntityAllocator.getCount(): " << mAllocators.mEntityAllocator.getCount()
              << " mEntityAllocator.getCapacity(): " << mAllocators.mEntityAllocator.getCapacity() << std::endl;
    std::cout << "mTransformAllocator.getCount(): " << mAllocators.mTransformAllocator.getCount()
              << " mTransformAllocator.getCapacity(): " << mAllocators.mTransformAllocator.getCapacity() << std::endl;
    std::cout << "mTransformDataAllocator.getCount(): " << mAllocators.mTransformDataAllocator.getCount()
              << " mTransformDataAllocator.getCapacity(): " << mAllocators.mTransformDataAllocator.getCapacity()
              << std::endl;
    mAllocators.mEntityAllocator.visualize();
    mAllocators.mTransformAllocator.visualize();
    mAllocators.mTransformDataAllocator.visualize();

    // immediate destroy all entities (and thereby also all components)
    std::vector<Guid> entitiesToDestroy;
    std::vector<Guid> entitiesToKeep;
    for (size_t i = 0; i < getNumberOfEntities(); i++)
    {
        Entity *entity = getEntityByIndex(i);

        if (entity->mDoNotDestroy)
        {
            entitiesToKeep.push_back(entity->getGuid());
        }
        else
        {
            entitiesToDestroy.push_back(entity->getGuid());
        }
    }

    for (size_t i = 0; i < entitiesToDestroy.size(); i++)
    {
        Log::info(("Immediate destroy entity with id: " + entitiesToDestroy[i].toString() + "\n").c_str());
        immediateDestroyEntity(entitiesToDestroy[i]);
    }

    assert(mAllocators.mEntityAllocator.getCount() == mAllocators.mTransformAllocator.getCount());
    assert(mAllocators.mEntityAllocator.getCount() == mAllocators.mTransformDataAllocator.getCount());

    std::cout << "After immediate destroy" << std::endl;
    std::cout << "mEntityAllocator.getCount(): " << mAllocators.mEntityAllocator.getCount()
              << " mEntityAllocator.getCapacity(): " << mAllocators.mEntityAllocator.getCapacity() << std::endl;
    std::cout << "mTransformAllocator.getCount(): " << mAllocators.mTransformAllocator.getCount()
              << " mTransformAllocator.getCapacity(): " << mAllocators.mTransformAllocator.getCapacity() << std::endl;
    std::cout << "mTransformDataAllocator.getCount(): " << mAllocators.mTransformDataAllocator.getCount()
              << " mTransformDataAllocator.getCapacity(): " << mAllocators.mTransformDataAllocator.getCapacity()
              << std::endl;
    mAllocators.mEntityAllocator.visualize();
    mAllocators.mTransformAllocator.visualize();
    mAllocators.mTransformDataAllocator.visualize();
    
    for (size_t i = 0; i < entitiesToKeep.size(); i++)
    {
        Log::info(("Keeping entity with id: " + entitiesToKeep[i].toString() + "\n").c_str());

    }
}

Entity *Scene::createEntity()
{
    // Create entity
    Entity* entity = this->addEntity();

    // Add transform (all entities must have a transform)
    this->addTransform(entity->getGuid());
   
    return entity;
}

// Used by scene loading
// currently create entity without transform...pass in transform node as well??
Entity *Scene::createEntity(const YAML::Node &in)
{
    return addEntity(in);
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
    
    int transformCount = 0;
    for (size_t i = 0; i < componentsOnEntity.size(); i++)
    {
        if (componentsOnEntity[i].second == ComponentType<Transform>::type)
        {
            transformCount++;
        }
        immediateDestroyComponent(entityGuid, componentsOnEntity[i].first, componentsOnEntity[i].second);
    }

    assert(transformCount == 1);

    removeEntity(entityGuid);

    assert(mAllocators.mEntityAllocator.getCount() == mAllocators.mTransformAllocator.getCount());
    assert(mAllocators.mEntityAllocator.getCount() == mAllocators.mTransformDataAllocator.getCount());
}

void Scene::latentDestroyComponent(const Guid &entityGuid, const Guid &componentGuid, int componentType)
{
    mIdState.mComponentGuidsMarkedLatentDestroy.push_back(std::make_tuple(entityGuid, componentGuid, componentType));
}

void Scene::immediateDestroyComponent(const Guid &entityGuid, const Guid &componentGuid, int componentType)
{
    // Destroy component
    switch (componentType)
    {
    case ComponentType<Transform>::type: {
        removeTransform(entityGuid, componentGuid);
        break;
    }
    case ComponentType<Rigidbody>::type: {
        removeRigidbody(entityGuid, componentGuid);
        break;
    }
    case ComponentType<Camera>::type: {
        removeCamera(entityGuid, componentGuid);
        break;
    }
    case ComponentType<MeshRenderer>::type: {
        removeMeshRenderer(entityGuid, componentGuid);
        break;
    }
    case ComponentType<Light>::type: {
        removeLight(entityGuid, componentGuid);
        break;
    }
    case ComponentType<BoxCollider>::type: {
        removeBoxCollider(entityGuid, componentGuid);
        break;
    }
    case ComponentType<SphereCollider>::type: {
        removeSphereCollider(entityGuid, componentGuid);
        break;
    }
    case ComponentType<Terrain>::type: {
        removeTerrain(entityGuid, componentGuid);
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

size_t Scene::getTransformDataCount() const
{
    return mAllocators.mTransformDataAllocator.getCount();
}

TransformData *Scene::getTransformDataByIndex(size_t index) const
{
    return mAllocators.mTransformDataAllocator.get(index);
}

TransformData *Scene::getTransformDataFromTransformId(const Id& id) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mTransformIdToGlobalIndex.find(id);
    return (it != mIdState.mTransformIdToGlobalIndex.end()) ? mAllocators.mTransformDataAllocator.get(it->second)
                                                            : nullptr;
}

TransformData *Scene::getTransformDataFromTransformGuid(const Guid &guid) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mTransformGuidToGlobalIndex.find(guid);
    return (it != mIdState.mTransformGuidToGlobalIndex.end()) ? mAllocators.mTransformDataAllocator.get(it->second)
                                                            : nullptr;
}

void Scene::setTransformPosition(const Id &id, const glm::vec3 &position)
{
    TransformData *transformData = this->getTransformDataFromTransformId(id);
    transformData->mPosition = position;
}

void Scene::setTransformRotation(const Id &id, const glm::quat &rotation)
{
    TransformData *transformData = this->getTransformDataFromTransformId(id);
    transformData->mRotation = rotation;
}

void Scene::setTransformScale(const Id &id, const glm::vec3 &scale)
{
    TransformData *transformData = this->getTransformDataFromTransformId(id);
    transformData->mScale = scale;
}

glm::vec3 Scene::getTransformPosition(const Id &id) const
{
    TransformData *transformData = this->getTransformDataFromTransformId(id);
    return transformData->mPosition;
}

glm::quat Scene::getTransformRotation(const Id &id) const
{
    TransformData *transformData = this->getTransformDataFromTransformId(id);
    return transformData->mRotation;
}

glm::vec3 Scene::getTransformScale(const Id &id) const
{
    TransformData *transformData = this->getTransformDataFromTransformId(id);
    return transformData->mScale;
}

glm::mat4 Scene::getTransformModelMatrix(const Id &id) const
{
    TransformData *transformData = this->getTransformDataFromTransformId(id);
    return transformData->getModelMatrix();
}

glm::vec3 Scene::getTransformForward(const Id &id) const
{
    TransformData *transformData = this->getTransformDataFromTransformId(id);
    return transformData->getForward();
}

glm::vec3 Scene::getTransformUp(const Id &id) const
{
    TransformData *transformData = this->getTransformDataFromTransformId(id);
    return transformData->getUp();
}

glm::vec3 Scene::getTransformRight(const Id &id) const
{
    TransformData *transformData = this->getTransformDataFromTransformId(id);
    return transformData->getRight();
}











size_t Scene::getIndexOfTransformFromMeshRendererIndex(size_t index) const
{
    return *mAllocators.mTransformIndicesAllocator.get(index);
}

TransformData *Scene::getTransformDataByMeshRendererIndex(size_t index) const
{
    return getTransformDataByIndex(*mAllocators.mTransformIndicesAllocator.get(index));
}












void Scene::addEntityToIdState(int entityIndex, const Guid &entityGuid, const Id &entityId)
{
    assert(entityGuid != Guid::INVALID);
    assert(entityId != Id::INVALID);
    assert(entityIndex >= 0);

    mIdState.mEntityGuidToGlobalIndex[entityGuid] = entityIndex;
    mIdState.mEntityIdToGlobalIndex[entityId] = entityIndex;

    mIdState.mGuidToGlobalIndex[entityGuid] = entityIndex;
    mIdState.mIdToGlobalIndex[entityId] = entityIndex;

    mIdState.mGuidToType[entityGuid] = EntityType<Entity>::type;
    mIdState.mIdToType[entityId] = EntityType<Entity>::type;

    mIdState.mGuidToId[entityGuid] = entityId;
    mIdState.mIdToGuid[entityId] = entityGuid;

    mIdState.mEntityGuidToComponentIds[entityGuid] = std::vector<std::pair<Guid, int>>();
    
    mIdState.mEntityGuidsMarkedCreated.push_back(entityGuid);
}

void Scene::addComponentToIdState(std::unordered_map<Guid, int> &componentGuidToIndex,
                                  std::unordered_map<Id, int> &componentIdToIndex, int componentIndex, const Guid &entityGuid,
                                  const Guid &componentGuid, const Id &componentId, int componentType)
{
    assert(componentGuid != Guid::INVALID);
    assert(componentId != Id::INVALID);
    assert(componentIndex >= 0);

    componentGuidToIndex[componentGuid] = componentIndex;
    componentIdToIndex[componentId] = componentIndex;

    mIdState.mGuidToGlobalIndex[componentGuid] = componentIndex;
    mIdState.mIdToGlobalIndex[componentId] = componentIndex;

    mIdState.mGuidToType[componentGuid] = componentType;
    mIdState.mIdToType[componentId] = componentType;

    mIdState.mGuidToId[componentGuid] = componentId;
    mIdState.mIdToGuid[componentId] = componentGuid;

    mIdState.mEntityGuidToComponentIds[entityGuid].push_back(std::make_pair(componentGuid, componentType));
    
    mIdState.mComponentGuidsMarkedCreated.push_back(std::make_tuple(entityGuid, componentGuid, componentType));
}

void Scene::removeEntityFromIdState(const Guid &entityGuid, const Id &entityId)
{
    assert(entityGuid != Guid::INVALID);
    assert(entityId != Id::INVALID);

    mIdState.mEntityGuidToGlobalIndex.erase(entityGuid);
    mIdState.mEntityIdToGlobalIndex.erase(entityId);

    mIdState.mGuidToGlobalIndex.erase(entityGuid);
    mIdState.mIdToGlobalIndex.erase(entityId);

    mIdState.mGuidToType.erase(entityGuid);
    mIdState.mIdToType.erase(entityId);

    mIdState.mGuidToId.erase(entityGuid);
    mIdState.mIdToGuid.erase(entityId);

    assert(mIdState.mEntityGuidToComponentIds[entityGuid].size() == 0);

    mIdState.mEntityGuidToComponentIds.erase(entityGuid);
}

void Scene::removeComponentFromIdState(std::unordered_map<Guid, int> &componentGuidToIndex,
                                       std::unordered_map<Id, int> &componentIdToIndex,
                                       const Guid &entityGuid, const Guid &componentGuid,
                                       const Id &componentId, int componentType)
{
    assert(componentGuid != Guid::INVALID);
    assert(componentId != Id::INVALID);

    componentGuidToIndex.erase(componentGuid);
    componentIdToIndex.erase(componentId);

    mIdState.mGuidToGlobalIndex.erase(componentGuid);
    mIdState.mIdToGlobalIndex.erase(componentId);

    mIdState.mGuidToType.erase(componentGuid);
    mIdState.mIdToType.erase(componentId);

    mIdState.mGuidToId.erase(componentGuid);
    mIdState.mIdToGuid.erase(componentId);

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
}

void Scene::moveEntityIndexInIdState(const Guid &entityGuid, const Id &entityId, int entityIndex)
{
    assert(entityGuid != Guid::INVALID);
    assert(entityId != Id::INVALID);
    assert(entityIndex >= 0);

    mIdState.mEntityGuidToGlobalIndex[entityGuid] = entityIndex;
    mIdState.mEntityIdToGlobalIndex[entityId] = entityIndex;

    mIdState.mGuidToGlobalIndex[entityGuid] = entityIndex;
    mIdState.mIdToGlobalIndex[entityId] = entityIndex;
}

void Scene::moveComponentIndexInIdState(std::unordered_map<Guid, int> &componentGuidToIndex,
                                        std::unordered_map<Id, int> &componentIdToIndex, const Guid &componentGuid,
                                        const Id &componentId, int componentIndex)
{
    assert(componentGuid != Guid::INVALID);
    assert(componentId != Id::INVALID);
    assert(componentIndex >= 0);

    componentGuidToIndex[componentGuid] = componentIndex;
    componentIdToIndex[componentId] = componentIndex;

    mIdState.mGuidToGlobalIndex[componentGuid] = componentIndex;
    mIdState.mIdToGlobalIndex[componentId] = componentIndex;
}

Entity *Scene::addEntity(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mEntityAllocator.getCount();

    Entity *entity = mAllocators.mEntityAllocator.construct(mWorld, Id::newId());
    entity->deserialize(in);

    addEntityToIdState(globalIndex, entity->getGuid(), entity->getId());

    return entity;
}

Transform *Scene::addTransform(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mTransformAllocator.getCount();

    Transform *transform = mAllocators.mTransformAllocator.construct(mWorld, Id::newId());
    transform->deserialize(in);

    TransformData *transformData = mAllocators.mTransformDataAllocator.construct();
    transformData->deserialize(in);

    assert(mAllocators.mEntityAllocator.getCount() == mAllocators.mTransformAllocator.getCount());
    assert(mAllocators.mEntityAllocator.getCount() == mAllocators.mTransformDataAllocator.getCount());

    addComponentToIdState(mIdState.mTransformGuidToGlobalIndex, 
                          mIdState.mTransformIdToGlobalIndex,
                          globalIndex,
                          transform->getEntityGuid(),
                          transform->getGuid(),
                          transform->getId(),
                          ComponentType<Transform>::type);
    return transform;
}

Rigidbody *Scene::addRigidbody(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mRigidbodyAllocator.getCount();

    Rigidbody *rigidbody = mAllocators.mRigidbodyAllocator.construct(mWorld, Id::newId());
    rigidbody->deserialize(in);

    addComponentToIdState(mIdState.mRigidbodyGuidToGlobalIndex, mIdState.mRigidbodyIdToGlobalIndex, globalIndex,
                          rigidbody->getEntityGuid(), rigidbody->getGuid(), rigidbody->getId(),
                          ComponentType<Rigidbody>::type);
    return rigidbody;
}

MeshRenderer *Scene::addMeshRenderer(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mMeshRendererAllocator.getCount();

    MeshRenderer *renderer = mAllocators.mMeshRendererAllocator.construct(mWorld, Id::newId());
    renderer->deserialize(in);

    addComponentToIdState(mIdState.mMeshRendererGuidToGlobalIndex, mIdState.mMeshRendererIdToGlobalIndex, globalIndex,
                          renderer->getEntityGuid(), renderer->getGuid(), renderer->getId(),
                          ComponentType<MeshRenderer>::type);

    // using renderer->getComponent<Transform>() requires the mActiveScene to be set already...
    //Transform *transform = renderer->getComponent<Transform>();
    Transform *transform = this->getComponent<Transform>(renderer->getEntityGuid());
    assert(transform != nullptr);

    size_t* index = mAllocators.mTransformIndicesAllocator.construct();
    *index = getIndexOf(transform->getId());

    return renderer;
}

Light *Scene::addLight(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mLightAllocator.getCount();

    Light *light = mAllocators.mLightAllocator.construct(mWorld, Id::newId());
    light->deserialize(in);

    addComponentToIdState(mIdState.mLightGuidToGlobalIndex, mIdState.mLightIdToGlobalIndex, globalIndex,
                          light->getEntityGuid(), light->getGuid(), light->getId(),
                          ComponentType<Light>::type);
    return light;
}

Camera *Scene::addCamera(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mCameraAllocator.getCount();

    Camera *camera = mAllocators.mCameraAllocator.construct(mWorld, Id::newId());
    camera->deserialize(in);

    addComponentToIdState(mIdState.mCameraGuidToGlobalIndex, mIdState.mCameraIdToGlobalIndex, globalIndex,
                          camera->getEntityGuid(), camera->getGuid(), camera->getId(), ComponentType<Camera>::type);
    return camera;
}

SphereCollider *Scene::addSphereCollider(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mSphereColliderAllocator.getCount();

    SphereCollider *collider = mAllocators.mSphereColliderAllocator.construct(mWorld, Id::newId());
    collider->deserialize(in);

    addComponentToIdState(mIdState.mSphereColliderGuidToGlobalIndex, mIdState.mSphereColliderIdToGlobalIndex,
                          globalIndex, collider->getEntityGuid(), collider->getGuid(), collider->getId(),
                          ComponentType<SphereCollider>::type);
    return collider;
}

BoxCollider *Scene::addBoxCollider(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mBoxColliderAllocator.getCount();

    BoxCollider *collider = mAllocators.mBoxColliderAllocator.construct(mWorld, Id::newId());
    collider->deserialize(in);

    addComponentToIdState(mIdState.mBoxColliderGuidToGlobalIndex, mIdState.mBoxColliderIdToGlobalIndex,
                          globalIndex, collider->getEntityGuid(), collider->getGuid(), collider->getId(),
                          ComponentType<BoxCollider>::type);
    return collider;
}

Terrain *Scene::addTerrain(const YAML::Node &in)
{
    int globalIndex = (int)mAllocators.mTerrainAllocator.getCount();

    Terrain *terrain = mAllocators.mTerrainAllocator.construct(mWorld, Id::newId());
    terrain->deserialize(in);

    addComponentToIdState(mIdState.mTerrainGuidToGlobalIndex, mIdState.mTerrainIdToGlobalIndex, globalIndex,
                          terrain->getEntityGuid(), terrain->getGuid(), terrain->getId(),
                          ComponentType<Terrain>::type);
    return terrain;
}

Entity *Scene::addEntity()
{
    int globalIndex = (int)mAllocators.mEntityAllocator.getCount();

    Entity *entity = mAllocators.mEntityAllocator.construct(mWorld, Guid::newGuid(), Id::newId());

    addEntityToIdState(globalIndex, entity->getGuid(), entity->getId());
    return entity;
}

Transform *Scene::addTransform(const Guid &entityGuid)
{
    int globalIndex = (int)mAllocators.mTransformAllocator.getCount();

    Transform *transform = mAllocators.mTransformAllocator.construct(mWorld, Guid::newGuid(), Id::newId());
    TransformData *transformData = mAllocators.mTransformDataAllocator.construct();

    assert(mAllocators.mEntityAllocator.getCount() == mAllocators.mTransformAllocator.getCount());
    assert(mAllocators.mEntityAllocator.getCount() == mAllocators.mTransformDataAllocator.getCount());

    assert(transform != nullptr);
    assert(transformData != nullptr);

    transform->mEntityGuid = entityGuid;

    addComponentToIdState(mIdState.mTransformGuidToGlobalIndex, mIdState.mTransformIdToGlobalIndex, globalIndex,
                          transform->getEntityGuid(), transform->getGuid(), transform->getId(),
                          ComponentType<Transform>::type);
    return transform;
}

Rigidbody *Scene::addRigidbody(const Guid &entityGuid)
{
    int globalIndex = (int)mAllocators.mRigidbodyAllocator.getCount();

    Rigidbody *rigidbody = mAllocators.mRigidbodyAllocator.construct(mWorld, Guid::newGuid(), Id::newId());

    rigidbody->mEntityGuid = entityGuid;

    addComponentToIdState(mIdState.mRigidbodyGuidToGlobalIndex, mIdState.mRigidbodyIdToGlobalIndex, globalIndex,
                          rigidbody->getEntityGuid(), rigidbody->getGuid(), rigidbody->getId(),
                          ComponentType<Rigidbody>::type);
    return rigidbody;
}

MeshRenderer *Scene::addMeshRenderer(const Guid &entityGuid)
{
    int globalIndex = (int)mAllocators.mMeshRendererAllocator.getCount();

    MeshRenderer *renderer = mAllocators.mMeshRendererAllocator.construct(mWorld, Guid::newGuid(), Id::newId());

    renderer->mEntityGuid = entityGuid;

    addComponentToIdState(mIdState.mMeshRendererGuidToGlobalIndex, mIdState.mMeshRendererIdToGlobalIndex, globalIndex,
                          renderer->getEntityGuid(), renderer->getGuid(), renderer->getId(),
                          ComponentType<MeshRenderer>::type);

    Transform *transform = renderer->getComponent<Transform>();
    assert(transform != nullptr);

    size_t *index = mAllocators.mTransformIndicesAllocator.construct();
    *index = getIndexOf(transform->getId());

    return renderer;
}

Light *Scene::addLight(const Guid &entityGuid)
{
    int globalIndex = (int)mAllocators.mLightAllocator.getCount();

    Light *light = mAllocators.mLightAllocator.construct(mWorld, Guid::newGuid(), Id::newId());

    light->mEntityGuid = entityGuid;

    addComponentToIdState(mIdState.mLightGuidToGlobalIndex, mIdState.mLightIdToGlobalIndex, globalIndex,
                          light->getEntityGuid(), light->getGuid(), light->getId(),
                          ComponentType<Light>::type);
    return light;
}

Camera *Scene::addCamera(const Guid &entityGuid)
{
    int globalIndex = (int)mAllocators.mCameraAllocator.getCount();

    Camera *camera = mAllocators.mCameraAllocator.construct(mWorld, Guid::newGuid(), Id::newId());

    camera->mEntityGuid = entityGuid;

    addComponentToIdState(mIdState.mCameraGuidToGlobalIndex, mIdState.mCameraIdToGlobalIndex, globalIndex,
                          camera->getEntityGuid(), camera->getGuid(), camera->getId(), ComponentType<Camera>::type);
    return camera;
}

SphereCollider *Scene::addSphereCollider(const Guid &entityGuid)
{
    int globalIndex = (int)mAllocators.mSphereColliderAllocator.getCount();

    SphereCollider *collider = mAllocators.mSphereColliderAllocator.construct(mWorld, Guid::newGuid(), Id::newId());

    collider->mEntityGuid = entityGuid;

    addComponentToIdState(mIdState.mSphereColliderGuidToGlobalIndex, mIdState.mSphereColliderIdToGlobalIndex, globalIndex,
                          collider->getEntityGuid(), collider->getGuid(), collider->getId(), ComponentType<SphereCollider>::type);
    return collider;
}

BoxCollider *Scene::addBoxCollider(const Guid &entityGuid)
{
    int globalIndex = (int)mAllocators.mBoxColliderAllocator.getCount();

    BoxCollider *collider = mAllocators.mBoxColliderAllocator.construct(mWorld, Guid::newGuid(), Id::newId());

    collider->mEntityGuid = entityGuid;

    addComponentToIdState(mIdState.mBoxColliderGuidToGlobalIndex, mIdState.mBoxColliderIdToGlobalIndex,
                          globalIndex, collider->getEntityGuid(), collider->getGuid(), collider->getId(),
                          ComponentType<BoxCollider>::type);
    return collider;
}

Terrain *Scene::addTerrain(const Guid &entityGuid)
{
    int globalIndex = (int)mAllocators.mTerrainAllocator.getCount();

    Terrain *terrain = mAllocators.mTerrainAllocator.construct(mWorld, Guid::newGuid(), Id::newId());

    terrain->mEntityGuid = entityGuid;

    addComponentToIdState(mIdState.mTerrainGuidToGlobalIndex, mIdState.mTerrainIdToGlobalIndex, globalIndex,
                          terrain->getEntityGuid(), terrain->getGuid(), terrain->getId(),
                          ComponentType<Terrain>::type);
    return terrain;
}

void Scene::removeEntity(const Guid &entityGuid)
{
    int entityIndex = getIndexOf(entityGuid);
    Id entityId = getIdFromGuid(entityGuid);

    Entity *swap = mAllocators.mEntityAllocator.destruct(entityIndex);

    assert(mAllocators.mEntityAllocator.getCount() == mAllocators.mTransformAllocator.getCount());
    assert(mAllocators.mEntityAllocator.getCount() == mAllocators.mTransformDataAllocator.getCount());

    removeEntityFromIdState(entityGuid, entityId);

    if (swap != nullptr)
    {
        moveEntityIndexInIdState(swap->getGuid(), swap->getId(), entityIndex);
    }
}

void Scene::removeTransform(const Guid &entityGuid, const Guid &componentGuid)
{
    int componentIndex = getIndexOf(componentGuid);
    Id componentId = getIdFromGuid(componentGuid);

    Transform *swap = mAllocators.mTransformAllocator.destruct(componentIndex);
    mAllocators.mTransformDataAllocator.destruct(componentIndex);

    assert(mAllocators.mTransformAllocator.getCount() == mAllocators.mTransformDataAllocator.getCount());

    assert(mAllocators.mEntityAllocator.getCount() - 1 == mAllocators.mTransformAllocator.getCount());
    assert(mAllocators.mEntityAllocator.getCount() - 1 == mAllocators.mTransformDataAllocator.getCount());

    removeComponentFromIdState(mIdState.mTransformGuidToGlobalIndex, mIdState.mTransformIdToGlobalIndex, entityGuid,
                               componentGuid, componentId, ComponentType<Transform>::type);

    if (swap != nullptr)
    {
        moveComponentIndexInIdState(mIdState.mTransformGuidToGlobalIndex, mIdState.mTransformIdToGlobalIndex,
                                    swap->getGuid(), swap->getId(), componentIndex);
    }
}

void Scene::removeRigidbody(const Guid &entityGuid, const Guid &componentGuid)
{
    int componentIndex = getIndexOf(componentGuid);
    Id componentId = getIdFromGuid(componentGuid);

    Rigidbody *swap = mAllocators.mRigidbodyAllocator.destruct(componentIndex);

    removeComponentFromIdState(mIdState.mRigidbodyGuidToGlobalIndex, mIdState.mRigidbodyIdToGlobalIndex, entityGuid,
                               componentGuid, componentId, ComponentType<Rigidbody>::type);

    if (swap != nullptr)
    {
        moveComponentIndexInIdState(mIdState.mRigidbodyGuidToGlobalIndex, mIdState.mRigidbodyIdToGlobalIndex,
                                    swap->getGuid(), swap->getId(), componentIndex);
    }
}

void Scene::removeMeshRenderer(const Guid &entityGuid, const Guid &componentGuid)
{
    int componentIndex = getIndexOf(componentGuid);
    Id componentId = getIdFromGuid(componentGuid);

    MeshRenderer *swap = mAllocators.mMeshRendererAllocator.destruct(componentIndex);
    mAllocators.mTransformIndicesAllocator.destruct(componentIndex);

    removeComponentFromIdState(mIdState.mMeshRendererGuidToGlobalIndex, mIdState.mMeshRendererIdToGlobalIndex,
                               entityGuid,
                               componentGuid, componentId, ComponentType<MeshRenderer>::type);

    if (swap != nullptr)
    {
        moveComponentIndexInIdState(mIdState.mMeshRendererGuidToGlobalIndex, mIdState.mMeshRendererIdToGlobalIndex,
                                    swap->getGuid(), swap->getId(), componentIndex);
    }
}

void Scene::removeLight(const Guid &entityGuid, const Guid &componentGuid)
{
    int componentIndex = getIndexOf(componentGuid);
    Id componentId = getIdFromGuid(componentGuid);

    Light *swap = mAllocators.mLightAllocator.destruct(componentIndex);

    removeComponentFromIdState(mIdState.mLightGuidToGlobalIndex, mIdState.mLightIdToGlobalIndex, entityGuid,
                               componentGuid, componentId, ComponentType<Light>::type);

    if (swap != nullptr)
    {
        moveComponentIndexInIdState(mIdState.mLightGuidToGlobalIndex, mIdState.mLightIdToGlobalIndex, swap->getGuid(),
                                    swap->getId(), componentIndex);
    }
}

void Scene::removeCamera(const Guid &entityGuid, const Guid &componentGuid)
{
    int componentIndex = getIndexOf(componentGuid);
    Id componentId = getIdFromGuid(componentGuid);

    Camera *swap = mAllocators.mCameraAllocator.destruct(componentIndex);

    removeComponentFromIdState(mIdState.mCameraGuidToGlobalIndex, mIdState.mCameraIdToGlobalIndex, entityGuid,
                               componentGuid, componentId, ComponentType<Camera>::type);

    if (swap != nullptr)
    {
        moveComponentIndexInIdState(mIdState.mCameraGuidToGlobalIndex, mIdState.mCameraIdToGlobalIndex, swap->getGuid(),
                                    swap->getId(), componentIndex);
    }
}

void Scene::removeSphereCollider(const Guid &entityGuid, const Guid &componentGuid)
{
    int componentIndex = getIndexOf(componentGuid);
    Id componentId = getIdFromGuid(componentGuid);

    SphereCollider *swap = mAllocators.mSphereColliderAllocator.destruct(componentIndex);

    removeComponentFromIdState(mIdState.mSphereColliderGuidToGlobalIndex, mIdState.mSphereColliderIdToGlobalIndex,
                               entityGuid, componentGuid, componentId, ComponentType<SphereCollider>::type);

    if (swap != nullptr)
    {
        moveComponentIndexInIdState(mIdState.mSphereColliderGuidToGlobalIndex, mIdState.mSphereColliderIdToGlobalIndex,
                                    swap->getGuid(), swap->getId(), componentIndex);
    }
}

void Scene::removeBoxCollider(const Guid &entityGuid, const Guid &componentGuid)
{
    int componentIndex = getIndexOf(componentGuid);
    Id componentId = getIdFromGuid(componentGuid);

    BoxCollider *swap = mAllocators.mBoxColliderAllocator.destruct(componentIndex);

    removeComponentFromIdState(mIdState.mBoxColliderGuidToGlobalIndex, mIdState.mBoxColliderIdToGlobalIndex, entityGuid,
                               componentGuid, componentId, ComponentType<BoxCollider>::type);

    if (swap != nullptr)
    {
        moveComponentIndexInIdState(mIdState.mBoxColliderGuidToGlobalIndex, mIdState.mBoxColliderIdToGlobalIndex,
                                    swap->getGuid(), swap->getId(), componentIndex);
    }
}

void Scene::removeTerrain(const Guid &entityGuid, const Guid &componentGuid)
{
    int componentIndex = getIndexOf(componentGuid);
    Id componentId = getIdFromGuid(componentGuid);

    Terrain *swap = mAllocators.mTerrainAllocator.destruct(componentIndex);

    removeComponentFromIdState(mIdState.mTerrainGuidToGlobalIndex, mIdState.mTerrainIdToGlobalIndex, entityGuid,
                               componentGuid, componentId, ComponentType<Terrain>::type);

    if (swap != nullptr)
    {
        moveComponentIndexInIdState(mIdState.mTerrainGuidToGlobalIndex, mIdState.mTerrainIdToGlobalIndex,
                                    swap->getGuid(), swap->getId(), componentIndex);
    }
}