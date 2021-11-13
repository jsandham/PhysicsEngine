#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_set>
#include <stack>

#include "../../include/core/Load.h"
#include "../../include/core/LoadInternal.h"
#include "../../include/core/Log.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

World::World()
{
    mPrimitives.createPrimitiveMeshes(this, 10, 10);
}

World::~World()
{
}

void World::loadAssetsInPath(const std::filesystem::path &filePath)
{
    if (std::filesystem::is_directory(filePath))
    {
        std::stack<std::filesystem::path> stack;
        stack.push(filePath);

        while (!stack.empty())
        {
            std::filesystem::path currentPath = stack.top();
            stack.pop();

            std::error_code error_code;
            for (const std::filesystem::directory_entry &entry :
                 std::filesystem::directory_iterator(currentPath, error_code))
            {
                if (std::filesystem::is_directory(entry, error_code))
                {
                    stack.push(entry.path());
                }
                else if (std::filesystem::is_regular_file(entry, error_code))
                {
                    std::string extension = entry.path().extension().string();
                    if (extension == ".mesh" || extension == ".shader" || extension == ".material" ||
                        extension == ".texture")
                    {
                        std::filesystem::path relativeDataPath =
                            entry.path().lexically_relative(std::filesystem::current_path());
                        loadAssetFromYAML(relativeDataPath.string());
                    }
                }
            }
        }
    }
}

Asset *World::loadAssetFromYAML(const std::string &filePath)
{
    YAML::Node in;
    try
    {
        in = YAML::LoadFile(filePath);
    }
    catch (YAML::Exception e /*YAML::BadFile e*/)
    {
        Log::error("YAML exception hit when trying to load file");
        return nullptr;
    }

    if (!in.IsMap() || in.begin() == in.end())
    {
        return nullptr;
    }

    if (in.begin()->first.IsScalar() && in.begin()->second.IsMap())
    {
        Asset *asset = loadAssetFromYAML(in.begin()->second);
        if (asset != nullptr)
        {
            mIdState.mAssetIdToFilepath[asset->getId()] = filePath;
            mIdState.mAssetFilepathToId[filePath] = asset->getId();
        }

        return asset;
    }

    return nullptr;
}

Scene *World::loadSceneFromYAML(const std::string &filePath)
{
    YAML::Node in;
    try
    {
        in = YAML::LoadFile(filePath);
    }
    catch (YAML::BadFile e)
    {
        Log::error("YAML exception hit when trying to load file");
        return nullptr;
    }

    Scene *scene = loadSceneFromYAML(in);
    if (scene != nullptr)
    {
        mIdState.mSceneIdToFilepath[scene->getId()] = filePath;
        mIdState.mSceneFilepathToId[filePath] = scene->getId();
    }

    return scene;
}

bool World::writeAssetToYAML(const std::string &filePath, const Guid &assetId) const
{
    std::ofstream out;
    out.open(filePath);

    if (!out.is_open())
    {
        std::string errorMessage = "Failed to open asset file " + filePath + "\n";
        Log::error(&errorMessage[0]);
        return false;
    }

    int type = getTypeOf(assetId);

    Asset *asset = nullptr;

    if (Asset::isInternal(type))
    {
        asset = PhysicsEngine::getInternalAsset(mAllocators, mIdState, assetId, type);
    }
    else
    {
        asset = PhysicsEngine::getAsset(mAllocators, mIdState, assetId, type);
    }

    if (asset->mHide == HideFlag::None)
    {
        YAML::Node an;
        asset->serialize(an);

        YAML::Node assetNode;
        assetNode[asset->getObjectName()] = an;

        out << assetNode;
        out << "\n";
    }

    out.close();

    return true;
}

bool World::writeSceneToYAML(const std::string &filePath, const Guid &sceneId) const
{
    std::ofstream out;
    out.open(filePath);

    if (!out.is_open())
    {
        std::string errorMessage = "Failed to open scene file " + filePath + "\n";
        Log::error(&errorMessage[0]);
        return false;
    }

    Scene *scene = getSceneById(sceneId);
    if (scene == nullptr)
    {
        return false;
    }

    YAML::Node sceneNode;
    scene->serialize(sceneNode);

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
                Component *component = nullptr;

                if (Component::isInternal(temp[j].second))
                {
                    component =
                        PhysicsEngine::getInternalComponent(mAllocators, mIdState, temp[j].first, temp[j].second);
                }
                else
                {
                    component = PhysicsEngine::getComponent(mAllocators, mIdState, temp[j].first, temp[j].second);
                }

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

Asset *World::loadAssetFromYAML(const YAML::Node &in)
{
    int type = YAML::getValue<int>(in, "type");
    Guid id = YAML::getValue<Guid>(in, "id");

    if (PhysicsEngine::isAsset(type) && id.isValid())
    {
        return loadAssetFromYAML(in, id, type);
    }

    return nullptr;
}

Scene *World::loadSceneFromYAML(const YAML::Node &in)
{
    int type = YAML::getValue<int>(in, "type");
    Guid id = YAML::getValue<Guid>(in, "id");

    if (PhysicsEngine::isScene(type) && id.isValid())
    {
        return loadSceneFromYAML(in, id);
    }

    return nullptr;
}

Asset *World::loadAssetFromYAML(const YAML::Node &in, const Guid id, int type)
{
    if (Asset::isInternal(type))
    {
        return PhysicsEngine::loadInternalAsset(*this, mAllocators, mIdState, in, id, type);
    }
    else
    {
        return PhysicsEngine::loadAsset(*this, mAllocators, mIdState, in, id, type);
    }
}

Scene *World::loadSceneFromYAML(const YAML::Node &in, const Guid id)
{
    return PhysicsEngine::loadInternalScene(*this, mAllocators, mIdState, in, id);
}

void World::latentDestroyEntitiesInWorld()
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

void World::immediateDestroyEntitiesInWorld()
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

size_t World::getNumberOfScenes() const
{
    return mAllocators.mSceneAllocator.getCount();
}

size_t World::getNumberOfEntities() const
{
    return mAllocators.mEntityAllocator.getCount();
}

size_t World::getNumberOfNonHiddenEntities() const
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

size_t World::getNumberOfUpdatingSystems() const
{
    return mSystems.size();
}

Mesh *World::getPrimtiveMesh(PrimitiveType type)
{
    switch (type)
    {
    case PrimitiveType::Plane:
        return getAssetById<Mesh>(mPrimitives.mPlaneMeshId);
    case PrimitiveType::Disc:
        return getAssetById<Mesh>(mPrimitives.mDiscMeshId);
    case PrimitiveType::Cube:
        return getAssetById<Mesh>(mPrimitives.mCubeMeshId);
    case PrimitiveType::Sphere:
        return getAssetById<Mesh>(mPrimitives.mSphereMeshId);
    case PrimitiveType::Cylinder:
        return getAssetById<Mesh>(mPrimitives.mCylinderMeshId);
    case PrimitiveType::Cone:
        return getAssetById<Mesh>(mPrimitives.mConeMeshId);
    default:
        return nullptr;
    }
}

Entity *World::createPrimitive(PrimitiveType type)
{
    Entity *entity = createEntity();
    Transform* transform = entity->addComponent<Transform>();
    MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>();
    
    assert(entity != nullptr);
    assert(transform != nullptr);
    assert(meshRenderer != nullptr);

    switch (type)
    {
    case PrimitiveType::Plane:
        entity->setName("Plane");
        break;
    case PrimitiveType::Disc:
        entity->setName("Disc");
        break;
    case PrimitiveType::Cube:
        entity->setName("Cube");
        break;
    case PrimitiveType::Sphere:
        entity->setName("Sphere");
        break;
    case PrimitiveType::Cylinder:
        entity->setName("Cylinder");
        break;
    case PrimitiveType::Cone:
        entity->setName("Cone");
        break;
    }

    transform->mPosition = glm::vec3(0, 0, 0);
    transform->mRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    transform->mScale = glm::vec3(1, 1, 1);
    meshRenderer->setMesh(getPrimtiveMesh(type)->getId());
    meshRenderer->setMaterial(mPrimitives.mStandardMaterialId);

    return entity;
}

Scene *World::getSceneById(const Guid &sceneId) const
{
    return getById_impl<Scene>(mIdState.mSceneIdToGlobalIndex, &mAllocators.mSceneAllocator, sceneId);
}

Scene *World::getSceneByIndex(size_t index) const
{
    return mAllocators.mSceneAllocator.get(index);
}

Entity *World::getEntityById(const Guid &entityId) const
{
    return getById_impl<Entity>(mIdState.mEntityIdToGlobalIndex, &mAllocators.mEntityAllocator, entityId);
}

Entity *World::getEntityByIndex(size_t index) const
{
    return mAllocators.mEntityAllocator.get(index);
}

System *World::getSystemByUpdateOrder(size_t order) const
{
    if (order >= mSystems.size())
    {
        return nullptr;
    }

    return mSystems[order];
}

int World::getIndexOf(const Guid &id) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mIdToGlobalIndex.find(id);
    if (it != mIdState.mIdToGlobalIndex.end())
    {
        return it->second;
    }

    return -1;
}

int World::getTypeOf(const Guid &id) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mIdToType.find(id);
    if (it != mIdState.mIdToType.end())
    {
        return it->second;
    }

    return -1;
}

Scene *World::createScene()
{
    int globalIndex = (int)mAllocators.mSceneAllocator.getCount();
    int type = SceneType<Scene>::type;
    Guid sceneId = Guid::newGuid();

    Scene *scene = mAllocators.mSceneAllocator.construct(this, sceneId);

    if (scene != nullptr)
    {
        addIdToGlobalIndexMap_impl<Scene>(scene->getId(), globalIndex, type);
    }

    return scene;
}

Entity *World::createEntity()
{
    int globalIndex = (int)mAllocators.mEntityAllocator.getCount();
    int type = EntityType<Entity>::type;
    Guid entityId = Guid::newGuid();

    Entity *entity = mAllocators.mEntityAllocator.construct(this, entityId);

    if (entity != nullptr)
    {
        addIdToGlobalIndexMap_impl<Entity>(entity->getId(), globalIndex, type);

        mIdState.mEntityIdToComponentIds[entityId] = std::vector<std::pair<Guid, int>>();

        mIdState.mEntityIdsMarkedCreated.push_back(entityId);
    }

    return entity;
}

void World::latentDestroyEntity(const Guid &entityId)
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

void World::immediateDestroyEntity(const Guid &entityId)
{
    std::vector<std::pair<Guid, int>> componentsOnEntity = mIdState.mEntityIdToComponentIds[entityId];
    for (size_t i = 0; i < componentsOnEntity.size(); i++)
    {
        immediateDestroyComponent(entityId, componentsOnEntity[i].first, componentsOnEntity[i].second);
    }

    assert(mIdState.mEntityIdToComponentIds[entityId].size() == 0);

    mIdState.mEntityIdToComponentIds.erase(entityId);

    destroyInternalEntity(mAllocators, mIdState, entityId, getIndexOf(entityId));
}

void World::latentDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType)
{
    mIdState.mComponentIdsMarkedLatentDestroy.push_back(std::make_tuple(entityId, componentId, componentType));
}

void World::immediateDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType)
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

    if (Component::isInternal(componentType))
    {
        destroyInternalComponent(mAllocators, mIdState, entityId, componentId, componentType, getIndexOf(componentId));
    }
    else
    {
        destroyComponent(mAllocators, mIdState, entityId, componentId, componentType, getIndexOf(componentId));
    }
}

void World::latentDestroyAsset(const Guid &assetId, int assetType)
{
    mIdState.mAssetIdsMarkedLatentDestroy.push_back(std::make_pair(assetId, assetType));
}

void World::immediateDestroyAsset(const Guid &assetId, int assetType)
{
    if (Asset::isInternal(assetType))
    {
        destroyInternalAsset(mAllocators, mIdState, assetId, assetType, getIndexOf(assetId));
    }
    else
    {
        destroyAsset(mAllocators, mIdState, assetId, assetType, getIndexOf(assetId));
    }
}

bool World::isMarkedForLatentDestroy(const Guid &id)
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

void World::clearIdsMarkedCreatedOrDestroyed()
{
    mIdState.mEntityIdsMarkedCreated.clear();
    mIdState.mEntityIdsMarkedLatentDestroy.clear();
    mIdState.mComponentIdsMarkedCreated.clear();
    mIdState.mComponentIdsMarkedLatentDestroy.clear();
}

std::vector<std::pair<Guid, int>> World::getComponentsOnEntity(const Guid &entityId) const
{
    std::vector<std::pair<Guid, int>> componentsOnEntity;

    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::const_iterator it =
        mIdState.mEntityIdToComponentIds.find(entityId);
    if (it != mIdState.mEntityIdToComponentIds.end())
    {
        componentsOnEntity = it->second;
    }

    return componentsOnEntity;
}

std::vector<Guid> World::getEntityIdsMarkedCreated() const
{
    return mIdState.mEntityIdsMarkedCreated;
}

std::vector<Guid> World::getEntityIdsMarkedLatentDestroy() const
{
    return mIdState.mEntityIdsMarkedLatentDestroy;
}

std::vector<std::tuple<Guid, Guid, int>> World::getComponentIdsMarkedCreated() const
{
    return mIdState.mComponentIdsMarkedCreated;
}

std::vector<std::tuple<Guid, Guid, int>> World::getComponentIdsMarkedLatentDestroy() const
{
    return mIdState.mComponentIdsMarkedLatentDestroy;
}

std::string World::getAssetFilepath(const Guid &assetId) const
{
    std::unordered_map<Guid, std::string>::const_iterator it = mIdState.mAssetIdToFilepath.find(assetId);
    if (it != mIdState.mAssetIdToFilepath.end())
    {
        return it->second;
    }

    return std::string();
}

std::string World::getSceneFilepath(const Guid &sceneId) const
{
    std::unordered_map<Guid, std::string>::const_iterator it = mIdState.mSceneIdToFilepath.find(sceneId);
    if (it != mIdState.mSceneIdToFilepath.end())
    {
        return it->second;
    }

    return std::string();
}

Guid World::getAssetId(const std::string& filepath) const
{
    std::unordered_map<std::string, Guid>::const_iterator it = mIdState.mAssetFilepathToId.find(filepath);
    if (it != mIdState.mAssetFilepathToId.end())
    {
        return it->second;
    }

    return Guid::INVALID;
}

Guid World::getSceneId(const std::string& filepath) const
{
    std::unordered_map<std::string, Guid>::const_iterator it = mIdState.mSceneFilepathToId.find(filepath);
    if (it != mIdState.mSceneFilepathToId.end())
    {
        return it->second;
    }

    return Guid::INVALID;
}

// bool World::raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance)
//{
//	Ray ray;
//
//	ray.origin = origin;
//	ray.direction = direction;
//
//	return sgrid.intersect(ray) != NULL;// || dtree.intersect(ray) != NULL;
//	// return stree.intersect(ray) != NULL || dtree.intersect(ray) != NULL;
//}
//
//// begin by only implementing for spheres first and later I will add for bounds, capsules etc
// bool World::raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance, Collider** collider)
//{
//	Ray ray;
//
//	ray.origin = origin;
//	ray.direction = direction;
//
//	// Object* object = stree.intersect(ray);
//	BoundingSphere* boundingSphere = sgrid.intersect(ray);
//
//	if(boundingSphere != NULL){
//		//std::cout << "AAAAAA id: " << boundingSphere->id.toString() << std::endl;
//		std::map<Guid, int>::iterator it = idToGlobalIndex.find(boundingSphere->id);
//		if(it != idToGlobalIndex.end()){
//			int colliderIndex = it->second;
//
//			if(boundingSphere->primitiveType == 0){
//				*collider = getComponentByIndex<SphereCollider>(colliderIndex);
//			}
//			else if(boundingSphere->primitiveType == 1){
//				*collider = getComponentByIndex<BoxCollider>(colliderIndex);
//			}
//			else{
//				*collider = getComponentByIndex<MeshCollider>(colliderIndex);
//			}
//			return true;
//		}
//		else{
//			std::cout << "Error: component id does not correspond to a global index" << std::endl;
//			return false;
//		}
//	}
//
//	return false;
//}