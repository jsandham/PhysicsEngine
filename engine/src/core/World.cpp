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
    mSceneId = Guid::INVALID;

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

Guid World::getSceneId() const
{
    return mSceneId;
}

bool World::loadAssetFromYAML(const std::string& filePath)
{
    YAML::Node in = YAML::LoadFile(filePath);

    if (!in.IsMap()) {
        return false;
    }

    for (YAML::const_iterator it = in.begin(); it != in.end(); ++it) {
        if (it->first.IsScalar() && it->second.IsMap()) {
            if (!loadYAML(it->second)) {
                return false;
            }
        }
    }

    return true;
}

bool World::loadSceneFromYAML(const std::string& filePath)
{
    YAML::Node in = YAML::LoadFile(filePath);

    if (!in.IsMap()) {
        return false;
    }

    if (in["id"])
    {
        mSceneId = in["id"].as<Guid>();
    }

    for (YAML::const_iterator it = in.begin(); it != in.end(); ++it) {
        if (it->first.IsScalar() && it->second.IsMap()) {
            if (!loadYAML(it->second)) {
                return false;
            }
        }
    }

    return true;
}

bool World::writeSceneToYAML(const std::string& filePath)
{
    std::ofstream out;
    out.open(filePath);

    if (!out.is_open()) {
        std::string errorMessage = "Failed to open scene file " + filePath + "\n";
        Log::error(&errorMessage[0]);
        return false;
    }

    for (size_t i = 0; i < getNumberOfEntities(); i++) {
        Entity* entity = getEntityByIndex(i);

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

            YAML::Node cn;
            component->serialize(cn);

            YAML::Node componentNode;
            componentNode[component->getObjectName()] = cn;

            out << componentNode;
            out << "\n";
        }
    }

    out.close();

    return true;
}

bool World::loadSceneFromBinary(const std::string &filePath)
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

    while(file.peek() != EOF)
    {
        ObjectHeader header;
        PhysicsEngine::read<ObjectHeader>(file, header);

        if (!loadBinary(file, header))
        {
            break;
        }
    }

    file.close();

    return true;
}

bool World::writeSceneToBinary(const std::string& filePath)
{
    std::ofstream file;
    file.open(filePath, std::ios::binary);

    if (!file.is_open())
    {
        std::string errorMessage = "Failed to open scene file " + filePath + "\n";
        Log::error(&errorMessage[0]);
        return false;
    }

    SceneHeader sceneHeader;
    sceneHeader.mSignature = SCENE_FILE_SIGNATURE;

    PhysicsEngine::write<SceneHeader>(file, sceneHeader);

    ObjectHeader header;
    for (size_t i = 0; i < getNumberOfEntities(); i++) {
        Entity* entity = getEntityByIndex(i);
        
        header.mId = entity->getId();
        header.mType = entity->getType();
        header.mIsTnternal = true;

        PhysicsEngine::write<ObjectHeader>(file, header);
        entity->serialize(file);

        std::vector<std::pair<Guid, int>> temp = entity->getComponentsOnEntity(this);
        for (size_t j = 0; j < temp.size(); j++)
        {
            Component* component = nullptr;

            if (Component::isInternal(temp[j].second))
            {
                component = PhysicsEngine::getInternalComponent(mAllocators, mIdState, temp[j].first, temp[j].second);
            }
            else
            {
                component = PhysicsEngine::getComponent(mAllocators, mIdState, temp[j].first, temp[j].second);
            }

            header.mId = component->getId();
            header.mType = component->getType();
            header.mIsTnternal = Component::isInternal(temp[j].second);

            PhysicsEngine::write<ObjectHeader>(file, header);

            component->serialize(file);
        }
    }

    return false;
}

bool World::loadBinary(std::ifstream& in, const ObjectHeader& header)
{
    if (PhysicsEngine::isEntity(header.mType))
    {
        loadEntityFromBinary(in, header);
    }
    else if (PhysicsEngine::isComponent(header.mType))
    {
        loadComponentFromBinary(in, header);
    }
    else if (PhysicsEngine::isSystem(header.mType))
    {
        loadSystemFromBinary(in, header);
    }
    else {
        return false;
    }

    return true;
}

void World::loadEntityFromBinary(std::ifstream &in, const ObjectHeader &header)
{
    if (header.mIsTnternal)
    {
        PhysicsEngine::loadInternalEntity(mAllocators, mIdState, in, header.mId);
    }
}

void World::loadComponentFromBinary(std::ifstream &in, const ObjectHeader &header)
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

void World::loadSystemFromBinary(std::ifstream &in, const ObjectHeader &header)
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

bool World::loadYAML(const YAML::Node& in)
{
    if (in["type"] && in["id"]) { //hasKey(const std::string& key)??
        int type = in["type"].as<int>(); //getValue<int>(const std::string& key)?? 
        Guid id = in["id"].as<Guid>();

        if (PhysicsEngine::isEntity(type))
        {
            loadEntityFromYAML(in, id);
        }
        else if (PhysicsEngine::isComponent(type))
        {
            loadComponentFromYAML(in, id, type);
        }
        else if (PhysicsEngine::isSystem(type))
        {
            loadSystemFromYAML(in, id, type);
        }
        else if (PhysicsEngine::isAsset(type))
        {
            loadAssetFromYAML(in, id, type);
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }

    return true;
}

void World::loadAssetFromYAML(const YAML::Node& in, const Guid id, int type)
{
    if(Asset::isInternal(type))
    {
        PhysicsEngine::loadInternalAsset(mAllocators, mIdState, in, id, type);
    }
    else
    {
        PhysicsEngine::loadAsset(mAllocators, mIdState, in, id, type);
    }
}

void World::loadEntityFromYAML(const YAML::Node& in, const Guid id)
{
    PhysicsEngine::loadInternalEntity(mAllocators, mIdState, in, id);
}

void World::loadComponentFromYAML(const YAML::Node& in, const Guid id, int type)
{
    if (Component::isInternal(type)) 
    {
        PhysicsEngine::loadInternalComponent(mAllocators, mIdState, in, id, type);
    }
    else
    {
        PhysicsEngine::loadComponent(mAllocators, mIdState, in, id, type);
    }
}

void World::loadSystemFromYAML(const YAML::Node& in, const Guid id, int type)
{
    if (System::isInternal(type)) 
    {
        PhysicsEngine::loadInternalSystem(mAllocators, mIdState, in, id, type);
    }
    else
    {
        PhysicsEngine::loadSystem(mAllocators, mIdState, in, id, type);
    }
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

Entity *World::getEntityByIndex(size_t index)
{
    return mAllocators.mEntityAllocator.get(index);
}

System *World::getSystemByUpdateOrder(size_t order)
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