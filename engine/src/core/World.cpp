#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_set>

#include "../../include/core/InternalMaterials.h"
#include "../../include/core/InternalMeshes.h"
#include "../../include/core/InternalShaders.h"
#include "../../include/core/Load.h"
#include "../../include/core/LoadInternal.h"
#include "../../include/core/Log.h"
#include "../../include/core/World.h"
#include "../../include/core/WorldSerialization.h"

using namespace PhysicsEngine;

World::World()
{
    // load default included meshes
    mDefaultAssets.mSphereMeshId = InternalMeshes::loadSphereMesh(this);
    mDefaultAssets.mCubeMeshId = InternalMeshes::loadCubeMesh(this);
    mDefaultAssets.mPlaneMeshId = InternalMeshes::loadPlaneMesh(this);

    // load default included shaders
    mDefaultAssets.mColorLitShaderId = InternalShaders::loadColorLitShader(this);
    mDefaultAssets.mNormalLitShaderId = InternalShaders::loadNormalLitShader(this);
    mDefaultAssets.mTangentLitShaderId = InternalShaders::loadTangentLitShader(this);

    mDefaultAssets.mFontShaderId = InternalShaders::loadFontShader(this);
    mDefaultAssets.mGizmoShaderId = InternalShaders::loadGizmoShader(this);
    mDefaultAssets.mLineShaderId = InternalShaders::loadLineShader(this);
    mDefaultAssets.mColorShaderId = InternalShaders::loadColorShader(this);
    mDefaultAssets.mPositionAndNormalsShaderId = InternalShaders::loadPositionAndNormalsShader(this);
    mDefaultAssets.mSsaoShaderId = InternalShaders::loadSsaoShader(this);
    mDefaultAssets.mScreenQuadShaderId = InternalShaders::loadScreenQuadShader(this);
    mDefaultAssets.mNormalMapShaderId = InternalShaders::loadNormalMapShader(this);
    mDefaultAssets.mDepthMapShaderId = InternalShaders::loadDepthMapShader(this);
    mDefaultAssets.mShadowDepthMapShaderId = InternalShaders::loadShadowDepthMapShader(this);
    mDefaultAssets.mShadowDepthCubemapShaderId = InternalShaders::loadShadowDepthCubemapShader(this);
    mDefaultAssets.mGbufferShaderId = InternalShaders::loadGBufferShader(this);
    mDefaultAssets.mSimpleLitShaderId = InternalShaders::loadSimpleLitShader(this);
    mDefaultAssets.mSimpleLitDeferedShaderId = InternalShaders::loadSimpleLitDeferredShader(this);
    mDefaultAssets.mOverdrawShaderId = InternalShaders::loadOverdrawShader(this);

    // load default included materials
    mDefaultAssets.mSimpleLitMaterialId =
        InternalMaterials::loadSimpleLitMaterial(this, mDefaultAssets.mSimpleLitShaderId);
    mDefaultAssets.mColorMaterialId = InternalMaterials::loadColorMaterial(this, mDefaultAssets.mColorShaderId);
}

World::~World()
{
}

Asset* World::loadAssetFromYAML(const std::string& filePath)
{
    YAML::Node in = YAML::LoadFile(filePath);

    if (!in.IsMap() || in.begin() == in.end()) {
        return nullptr;
    }

    if (in.begin()->first.IsScalar() && in.begin()->second.IsMap())
    {
        Asset* asset = loadAssetFromYAML(in.begin()->second);
        if (asset != nullptr)
        {
            mIdState.mAssetIdToFilepath[asset->getId()] = filePath;
        }

        return asset;
    }

    return nullptr;
}

Scene* World::loadSceneFromYAML(const std::string& filePath)
{
    YAML::Node in = YAML::LoadFile(filePath);

    Scene* scene = loadSceneFromYAML(in);
    if (scene != nullptr)
    {
        mIdState.mSceneIdToFilepath[scene->getId()] = filePath;
    }

    return scene;
}

bool World::writeSceneToYAML(const std::string& filePath, const Guid& sceneId) const
{
    std::ofstream out;
    out.open(filePath);

    if (!out.is_open()) {
        std::string errorMessage = "Failed to open scene file " + filePath + "\n";
        Log::error(&errorMessage[0]);
        return false;
    }

    Scene* scene = getSceneById(sceneId);
    if (scene == nullptr) {
        return false;
    }

    YAML::Node sceneNode;
    sceneNode["type"] = scene->getType();
    sceneNode["id"] = scene->getId();

    out << sceneNode;
    out << "\n";

    for (size_t i = 0; i < getNumberOfEntities(); i++) {
        const Entity* entity = getEntityByIndex(i);
        
        if (entity->mHide == HideFlag::None)
        {
            YAML::Node en;
            entity->serialize(en);

            YAML::Node entityNode;
            entityNode[entity->getObjectName()] = en;

            out << entityNode;
            out << "\n";

            std::vector<std::pair<Guid, int>> temp = entity->getComponentsOnEntity(this);
            for (size_t j = 0; j < temp.size(); j++) {
                Component* component = nullptr;

                if (Component::isInternal(temp[j].second))
                {
                    component = PhysicsEngine::getInternalComponent(mAllocators, mIdState, temp[j].first, temp[j].second);
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

Asset* World::loadAssetFromYAML(const YAML::Node& in)
{
    int type = YAML::getValue<int>(in, "type");
    Guid id = YAML::getValue<Guid>(in, "id");

    if (PhysicsEngine::isAsset(type) && id.isValid())
    {
        return loadAssetFromYAML(in, id, type);
    }

    return nullptr;
}

Scene* World::loadSceneFromYAML(const YAML::Node& in)
{
    int type = YAML::getValue<int>(in, "type");
    Guid id = YAML::getValue<Guid>(in, "id");

    if (PhysicsEngine::isScene(type) && id.isValid())
    {
        return loadSceneFromYAML(in, id);
    }

    return nullptr;
}

Asset* World::loadAssetFromYAML(const YAML::Node& in, const Guid id, int type)
{
    if(Asset::isInternal(type))
    {
        return PhysicsEngine::loadInternalAsset(mAllocators, mIdState, in, id, type);
    }
    else
    {
        return PhysicsEngine::loadAsset(mAllocators, mIdState, in, id, type);
    }
}

Scene* World::loadSceneFromYAML(const YAML::Node& in, const Guid id)
{
    return PhysicsEngine::loadInternalScene(mAllocators, mIdState, in, id);
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
        Entity* entity = getEntityByIndex(i);

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
        const Entity* entity = getEntityByIndex(i);
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

Scene* World::getSceneById(const Guid& sceneId) const
{
    return getById_impl<Scene>(mIdState.mSceneIdToGlobalIndex, &mAllocators.mSceneAllocator, sceneId);
}

Scene* World::getSceneByIndex(size_t index) const
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

Scene* World::createScene()
{
    int globalIndex = (int)mAllocators.mSceneAllocator.getCount();
    int type = SceneType<Scene>::type;
    Guid sceneId = Guid::newGuid();

    Scene* scene = mAllocators.mSceneAllocator.construct(sceneId);

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

    Entity *entity = mAllocators.mEntityAllocator.construct(entityId);

    if (entity != nullptr)
    {
        addIdToGlobalIndexMap_impl<Entity>(entity->getId(), globalIndex, type);

        mIdState.mEntityIdToComponentIds[entityId] = std::vector<std::pair<Guid, int>>();

        mIdState.mEntityIdsMarkedCreated.push_back(entityId);
    }

    return entity;
}

Entity *World::createEntity(std::istream &in)
{
    int globalIndex = (int)mAllocators.mEntityAllocator.getCount();
    int type = EntityType<Entity>::type;

    Entity *entity = mAllocators.mEntityAllocator.construct(in);

    if (entity != nullptr)
    {
        addIdToGlobalIndexMap_impl<Entity>(entity->getId(), globalIndex, type);

        mIdState.mEntityIdToComponentIds[entity->getId()] = std::vector<std::pair<Guid, int>>();

        mIdState.mEntityIdsMarkedCreated.push_back(entity->getId());
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
    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::const_iterator it =
        mIdState.mEntityIdToComponentIds.find(entityId);

    assert(it != mIdState.mEntityIdToComponentIds.end());

    for (size_t i = 0; i < it->second.size(); i++)
    {
        immediateDestroyComponent(entityId, it->second[i].first, it->second[i].second);
    }

    mIdState.mEntityIdToComponentIds.erase(it);

    destroyInternalEntity(mAllocators, mIdState, entityId, getIndexOf(entityId));
}

void World::latentDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType)
{
    mIdState.mComponentIdsMarkedLatentDestroy.push_back(std::make_tuple(entityId, componentId, componentType));
}

void World::immediateDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType)
{
    if (Component::isInternal(componentType))
    {
        destroyInternalComponent(mAllocators, mIdState, componentId, componentType, getIndexOf(componentId));
    }
    else
    {
        destroyComponent(mAllocators, mIdState, componentId, componentType, getIndexOf(componentId));
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

Guid World::getSphereMesh() const
{
    return mDefaultAssets.mSphereMeshId;
}

Guid World::getCubeMesh() const
{
    return mDefaultAssets.mCubeMeshId;
}

Guid World::getPlaneMesh() const
{
    return mDefaultAssets.mPlaneMeshId;
}

Guid World::getColorMaterial() const
{
    return mDefaultAssets.mColorMaterialId;
}

Guid World::getSimpleLitMaterial() const
{
    return mDefaultAssets.mSimpleLitMaterialId;
}

Guid World::getColorLitShaderId() const
{
    return mDefaultAssets.mColorLitShaderId;
}

Guid World::getNormalLitShaderId() const
{
    return mDefaultAssets.mNormalLitShaderId;
}

Guid World::getTangentLitShaderId() const
{
    return mDefaultAssets.mTangentLitShaderId;
}

Guid World::getFontShaderId() const
{
    return mDefaultAssets.mFontShaderId;
}

Guid World::getGizmoShaderId() const
{
    return mDefaultAssets.mGizmoShaderId;
}

Guid World::getLineShaderId() const
{
    return mDefaultAssets.mLineShaderId;
}

Guid World::getColorShaderId() const
{
    return mDefaultAssets.mColorShaderId;
}

Guid World::getPositionAndNormalsShaderId() const
{
    return mDefaultAssets.mPositionAndNormalsShaderId;
}

Guid World::getSsaoShaderId() const
{
    return mDefaultAssets.mSsaoShaderId;
}

Guid World::getScreenQuadShaderId() const
{
    return mDefaultAssets.mScreenQuadShaderId;
}

Guid World::getNormalMapShaderId() const
{
    return mDefaultAssets.mNormalMapShaderId;
}

Guid World::getDepthMapShaderId() const
{
    return mDefaultAssets.mDepthMapShaderId;
}

Guid World::getShadowDepthMapShaderId() const
{
    return mDefaultAssets.mShadowDepthMapShaderId;
}

Guid World::getShadowDepthCubemapShaderId() const
{
    return mDefaultAssets.mShadowDepthCubemapShaderId;
}

Guid World::getGbufferShaderId() const
{
    return mDefaultAssets.mGbufferShaderId;
}

Guid World::getSimpleLitShaderId() const
{
    return mDefaultAssets.mSimpleLitShaderId;
}

Guid World::getSimpleLitDeferredShaderId() const
{
    return mDefaultAssets.mSimpleLitDeferedShaderId;
}

Guid World::getOverdrawShaderId() const
{
    return mDefaultAssets.mOverdrawShaderId;
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