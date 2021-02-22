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

bool World::loadAsset(const std::string &filePath)
{
    std::ifstream file;
    file.open(filePath, std::ios::binary);

    if (!file.is_open())
    {
        std::string errorMessage = "Failed to open asset bundle " + filePath + "\n";
        Log::error(&errorMessage[0]);
        return false;
    }

    AssetHeader header;
    PhysicsEngine::read<AssetHeader>(file, header);

    assert(header.mSignature == ASSET_FILE_SIGNATURE && "Trying to load an invalid binary asset file\n");

    std::vector<ObjectHeader> assets(header.mAssetCount);

    for (size_t i = 0; i < assets.size(); i++)
    {
        PhysicsEngine::read<ObjectHeader>(file, assets[i]);
    }

    for (size_t i = 0; i < assets.size(); i++)
    {
        loadAsset(file, assets[i]);
    }

    file.close();

    return true;
}

bool World::loadScene(const std::string &filePath, bool ignoreSystemsAndCamera)
{
    std::ifstream file;
    file.open(filePath, std::ios::binary);

    if (!file.is_open())
    {
        std::string errorMessage = "Failed to open scene file " + filePath + "\n";
        Log::error(&errorMessage[0]);
        return false;
    }

    SceneHeader sceneHeader;
    PhysicsEngine::read<SceneHeader>(file, sceneHeader);

    assert(sceneHeader.mSignature == SCENE_FILE_SIGNATURE && "Trying to load an invalid binary scene file\n");

    std::vector<ObjectHeader> entities(sceneHeader.mEntityCount);
    std::vector<ObjectHeader> components(sceneHeader.mComponentCount);
    std::vector<ObjectHeader> systems(sceneHeader.mSystemCount);

    for (size_t i = 0; i < entities.size(); i++)
    {
        PhysicsEngine::read<ObjectHeader>(file, entities[i]);
    }

    for (size_t i = 0; i < components.size(); i++)
    {
        PhysicsEngine::read<ObjectHeader>(file, components[i]);
    }

    for (size_t i = 0; i < systems.size(); i++)
    {
        PhysicsEngine::read<ObjectHeader>(file, systems[i]);
    }

    for (size_t i = 0; i < entities.size(); i++)
    {
        loadEntity(file, entities[i]);
    }

    for (size_t i = 0; i < components.size(); i++)
    {
        loadComponent(file, components[i]);
    }

    for (size_t i = 0; i < systems.size(); i++)
    {
        loadSystem(file, systems[i]);
    }

    file.close();

    return true;
}

bool World::loadSceneFromEditor(const std::string &filePath)
{
    return loadScene(filePath, true);
}

void World::loadAsset(std::ifstream &in, const ObjectHeader &header)
{
    if (header.mIsTnternal)
    {
        PhysicsEngine::loadInternalAsset(mAllocators, mIdState, in, header.mId, header.mType);
    }
    else
    {
        PhysicsEngine::loadAsset(mAllocators, mIdState, in, header.mId, header.mType);
    }
}

void World::loadEntity(std::ifstream &in, const ObjectHeader &header)
{
    if (header.mIsTnternal)
    {
        PhysicsEngine::loadInternalEntity(mAllocators, mIdState, in, header.mId);
    }
}

void World::loadComponent(std::ifstream &in, const ObjectHeader &header)
{
    if (header.mIsTnternal)
    {
        PhysicsEngine::loadInternalComponent(mAllocators, mIdState, in, header.mId, header.mType);
    }
    else
    {
        PhysicsEngine::loadComponent(mAllocators, mIdState, in, header.mId, header.mType);
    }
}

void World::loadSystem(std::ifstream &in, const ObjectHeader &header)
{
    if (header.mIsTnternal)
    {
        PhysicsEngine::loadInternalSystem(mAllocators, mIdState, in, header.mId, header.mType);
    }
    else
    {
        PhysicsEngine::loadSystem(mAllocators, mIdState, in, header.mId, header.mType);
    }
}

void World::latentDestroyEntitiesInWorld()
{
    // latent destroy all entities (and thereby also all components)
    for (int i = 0; i < getNumberOfEntities(); i++)
    {
        Entity *entity = getEntityByIndex(i);

        if (!entity->mDoNotDestroy)
        {
            latentDestroyEntity(entity->getId());
        }
    }
}

size_t World::getNumberOfEntities() const
{
    return mAllocators.mEntityAllocator.getCount();
}

size_t World::getNumberOfUpdatingSystems() const
{
    return mSystems.size();
}

Entity *World::getEntityById(const Guid &entityId)
{
    return getById_impl<Entity>(mIdState.mEntityIdToGlobalIndex, &mAllocators.mEntityAllocator, entityId);
}

Entity *World::getEntityByIndex(int index)
{
    return mAllocators.mEntityAllocator.get(index);
}

System *World::getSystemByUpdateOrder(int order)
{
    if (order < 0 || order >= mSystems.size())
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

Entity *World::createEntity()
{
    int globalIndex = (int)mAllocators.mEntityAllocator.getCount();
    int type = EntityType<Entity>::type;
    Guid entityId = Guid::newGuid();

    Entity *entity = mAllocators.mEntityAllocator.construct(entityId);

    if (entity != nullptr)
    {
        entity->mDoNotDestroy = false;

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

    int index = getIndexOf(entityId);

    assert(index != -1);

    destroyInternalEntity(mAllocators, mIdState, entityId, index);
}

void World::latentDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType)
{
    mIdState.mComponentIdsMarkedLatentDestroy.push_back(std::make_tuple(entityId, componentId, componentType));
}

void World::immediateDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType)
{
    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::iterator it =
        mIdState.mEntityIdToComponentIds.find(entityId);

    assert(it != mIdState.mEntityIdToComponentIds.end());

    for (size_t i = 0; i < it->second.size(); i++)
    {
        if (it->second[i].first == componentId)
        {
            it->second.erase(it->second.begin() + i);
            break;
        }
    }

    int index = getIndexOf(componentId);

    assert(index != -1);

    if (Component::isInternal(componentType))
    {
        destroyInternalComponent(mAllocators, mIdState, componentId, componentType, index);
    }
    else
    {
        destroyComponent(mAllocators, mIdState, componentId, componentType, index);
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

std::vector<std::pair<Guid, int>> World::getComponentsOnEntity(const Guid &entityId)
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
    std::unordered_map<Guid, std::string>::const_iterator it = mAssetIdToFilepath.find(assetId);
    if (it != mAssetIdToFilepath.end())
    {
        return it->second;
    }

    return std::string();
}

std::string World::getSceneFilepath(const Guid &sceneId) const
{
    std::unordered_map<Guid, std::string>::const_iterator it = mSceneIdToFilepath.find(sceneId);
    if (it != mSceneIdToFilepath.end())
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