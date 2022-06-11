#include <fstream>

#include "../../include/core/Scene.h"
#include "../../include/core/GLM.h"
#include "../../include/core/Types.h"
#include "../../include/core/Version.h"
#include "../../include/core/Entity.h"
#include "../../include/core/Log.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

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