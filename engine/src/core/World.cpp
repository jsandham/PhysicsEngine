#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_set>

#include "../../include/core/Log.h"
#include "../../include/core/Load.h"
#include "../../include/core/LoadInternal.h"
#include "../../include/core/InternalMeshes.h"
#include "../../include/core/InternalShaders.h"
#include "../../include/core/InternalMaterials.h"
#include "../../include/core/World.h"
#include "../../include/core/Geometry.h"
#include "../../include/core/WorldSerialization.h"

using namespace PhysicsEngine;

World::World()
{
	// load default included meshes
	mSphereMeshId = InternalMeshes::loadSphereMesh(this);
	mCubeMeshId = InternalMeshes::loadCubeMesh(this);

	// load default included shaders
	mFontShaderId = InternalShaders::loadFontShader(this);
	mColorShaderId = InternalShaders::loadColorShader(this);
	mPositionAndNormalsShaderId = InternalShaders::loadPositionAndNormalsShader(this);
	mSsaoShaderId = InternalShaders::loadSsaoShader(this);
	mScreenQuadShaderId = InternalShaders::loadScreenQuadShader(this);
	mNormalMapShaderId = InternalShaders::loadNormalMapShader(this);
	mDepthMapShaderId = InternalShaders::loadDepthMapShader(this);
	mShadowDepthMapShaderId = InternalShaders::loadShadowDepthMapShader(this);
	mShadowDepthCubemapShaderId = InternalShaders::loadShadowDepthCubemapShader(this);
	mGbufferShaderId = InternalShaders::loadGBufferShader(this);
	mSimpleLitShaderId = InternalShaders::loadSimpleLitShader(this);
	mSimpleLitDeferedShaderId = InternalShaders::loadSimpleLitDeferredShader(this);
	mOverdrawShaderId = InternalShaders::loadOverdrawShader(this);

	// load default included materials
	mSimpleLitMaterialId = InternalMaterials::loadSimpleLitMaterial(this, mSimpleLitShaderId);
	mColorMaterialId = InternalMaterials::loadColorMaterial(this, mColorShaderId);
}

World::~World()
{
	for (auto it = mComponentAllocatorMap.begin(); it != mComponentAllocatorMap.end(); it++) {
		delete it->second;
	}

	for (auto it = mSystemAllocatorMap.begin(); it != mSystemAllocatorMap.end(); it++) {
		delete it->second;
	}

	for (auto it = mAssetAllocatorMap.begin(); it != mAssetAllocatorMap.end(); it++) {
		delete it->second;
	}
}

bool World::loadAsset(const std::string& filePath)
{
	PhysicsEngine::loadAssetIntoWorld(filePath, 
									mMeshAllocator,
									mMaterialAllocator,
									mShaderAllocator,
									mTexture2DAllocator,
									mTexture3DAllocator,
									mCubemapAllocator,
									mFontAllocator, 
									mAssetAllocatorMap,
									mIdToGlobalIndex, 
									mIdToType,
									mAssetIdToFilepath);

	return true;
}

bool World::loadScene(const std::string& filePath, bool ignoreSystemsAndCamera)
{
	PhysicsEngine::loadSceneIntoWorld(filePath,
		mEntityAllocator,
		mTransformAllocator,
		mMeshRendererAllocator,
		mLineRendererAllocator,
		mRigidbodyAllocator,
		mCameraAllocator,
		mLightAllocator,
		mSphereColliderAllocator,
		mBoxColliderAllocator,
		mCapsuleColliderAllocator,
		mMeshColliderAllocator,
		mRenderSystemAllocator,
		mPhysicsSystemAllocator,
		mCleanupSystemAllocator,
		mDebugSystemAllocator,
		mComponentAllocatorMap,
		mSystemAllocatorMap,
		mIdToGlobalIndex,
		mIdToType,
		mEntityIdToComponentIds,
		mEntityIdsMarkedCreated,
		mComponentIdsMarkedCreated,
		mSceneIdToFilepath);
	
	return true;
}

bool World::loadSceneFromEditor(const std::string& filePath)
{
	return loadScene(filePath, true);
}

void World::latentDestroyEntitiesInWorld() 
{
	// latent destroy all entities (and thereby also all components)
	for (int i = 0; i < getNumberOfEntities(); i++) {
		Entity* entity = getEntityByIndex(i);

		if (!entity->mDoNotDestroy) {
			latentDestroyEntity(entity->mEntityId);
		}
	}
}

int World::getNumberOfEntities() const
{
	return (int)mEntityAllocator.getCount();
}

int World::getNumberOfUpdatingSystems() const
{
	return (int)mSystems.size();
}

// error here if entityId does not correspond to an entity but instead an asset or component
Entity* World::getEntityById(const Guid& entityId)
{
	if (entityId == Guid::INVALID) {
		return NULL;
	}

	std::unordered_map<Guid, int>::iterator it = mIdToGlobalIndex.find(entityId);
	if(it != mIdToGlobalIndex.end()){
		return mEntityAllocator.get(it->second);
	}

	return NULL;
}

Entity* World::getEntityByIndex(int index)
{
	return mEntityAllocator.get(index);
}

System* World::getSystemByUpdateOrder(int order)
{
	if (order < 0 || order >= mSystems.size()) {
		return NULL;
	}

	return mSystems[order];
}

int World::getIndexOf(const Guid& id) const
{
	std::unordered_map<Guid, int>::const_iterator it = mIdToGlobalIndex.find(id);
	if( it != mIdToGlobalIndex.end()){
		return it->second;
	}

	return -1;
}

int World::getTypeOf(const Guid& id) const
{
	std::unordered_map<Guid, int>::const_iterator it = mIdToType.find(id);
	if (it != mIdToType.end()) {
		return it->second;
	}

	return -1;
}

Entity* World::createEntity()
{
	int globalIndex = (int)mEntityAllocator.getCount();
	int type = EntityType<Entity>::type;
	Guid entityId = Guid::newGuid();

	Entity* entity = mEntityAllocator.construct();

	if (entity != NULL) {
		entity->mEntityId = entityId;
		entity->mDoNotDestroy = false;

		mIdToGlobalIndex[entityId] = globalIndex;
		mIdToType[entityId] = type;

		mEntityIdToComponentIds[entityId] = std::vector<std::pair<Guid, int>>();

		mEntityIdsMarkedCreated.push_back(entityId);
	}

	return entity;
}

Entity* World::createEntity(const std::vector<char>& data)
{
	int globalIndex = (int)mEntityAllocator.getCount();
	int type = EntityType<Entity>::type;

	Entity* entity = mEntityAllocator.construct(data);

	if (entity != NULL) {
		mIdToGlobalIndex[entity->mEntityId] = globalIndex;
		mIdToType[entity->mEntityId] = type;

		mEntityIdToComponentIds[entity->mEntityId] = std::vector<std::pair<Guid, int>>();

		mEntityIdsMarkedCreated.push_back(entity->mEntityId);
	}

	return entity;
}

void World::latentDestroyEntity(const Guid& entityId)
{
	mEntityIdsMarkedLatentDestroy.push_back(entityId);

	// add any components found on the entity to the latent destroy component list
	std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::iterator it = mEntityIdToComponentIds.find(entityId);
	if(it != mEntityIdToComponentIds.end()){

		std::vector<std::pair<Guid, int>> componentsOnEntity = it->second;
		for(size_t i = 0; i < componentsOnEntity.size(); i++){
			Guid componentId = componentsOnEntity[i].first;
			int componentType = componentsOnEntity[i].second;

			latentDestroyComponent(entityId, componentId, componentType);
		}
	}
	else{
		std::string message = "Error: Could not find entity with id " + entityId.toString() + " when trying to add to latent destroy list\n";
		Log::error(message.c_str());
		return;
	}
}

void World::immediateDestroyEntity(const Guid& entityId)
{
	std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::iterator it1 = mEntityIdToComponentIds.find(entityId);
	if(it1 != mEntityIdToComponentIds.end()){
		std::vector<std::pair<Guid, int>> componentsOnEntity = it1->second;

		for(size_t i = 0; i < componentsOnEntity.size(); i++){
			Guid componentId = componentsOnEntity[i].first;
			int componentType = componentsOnEntity[i].second;

			immediateDestroyComponent(entityId, componentId, componentType);
		}

		mEntityIdToComponentIds.erase(it1);
	}
	else{
		std::string message = "Error: Could not find entity with id " + entityId.toString() + " when trying to delete\n";
		Log::error(message.c_str());
		return;
	}

	// remove from id to global index map
	std::unordered_map<Guid, int>::iterator it2 = mIdToGlobalIndex.find(entityId);
	if(it2 != mIdToGlobalIndex.end()){
		int index = it2->second;

		Entity* swappedEntity = destroyInternalEntity(&mEntityAllocator, index);

		mIdToGlobalIndex.erase(it2);

		if(swappedEntity != NULL){
			mEntityIdsMarkedMoved.push_back(std::make_pair(swappedEntity->mEntityId, mIdToGlobalIndex[swappedEntity->mEntityId]));

			mIdToGlobalIndex[swappedEntity->mEntityId] = index;
		}
	}
	else{
		std::string message = "Error: Could not find entity with id " + entityId.toString() + " when trying to delete from id to global index map\n";
		Log::error(message.c_str());
		return;
	}

	//remove from id to type map
	std::unordered_map<Guid, int>::iterator it3 = mIdToType.find(entityId);
	if (it3 != mIdToType.end()) {
		mIdToType.erase(it3);
	}
	else {
		std::string message = "Error: Could not find entity with id " + entityId.toString() + " when trying to delete from id to type map\n";
		Log::error(message.c_str());
		return;
	}
}

void World::latentDestroyComponent(const Guid& entityId, const Guid& componentId, int componentType)
{
	mComponentIdsMarkedLatentDestroy.push_back(make_triple(entityId, componentId, componentType));
}

void World::immediateDestroyComponent(const Guid& entityId, const Guid &componentId, int componentType)
{
	std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::iterator it1 = mEntityIdToComponentIds.find(entityId);
	if(it1 != mEntityIdToComponentIds.end()){
		for(size_t i = 0; i < it1->second.size(); i++){
			if(it1->second[i].first == componentId){
				it1->second.erase(it1->second.begin() + i);
				break;
			}
		}
	}

	// remove from id to global index map
	std::unordered_map<Guid, int>::iterator it2 = mIdToGlobalIndex.find(componentId);
	if(it2 != mIdToGlobalIndex.end()){
		int index = it2->second;

		Component* swappedComponent = NULL;
		if(componentType < 20){
			swappedComponent = destroyInternalComponent(&mTransformAllocator,
														&mMeshRendererAllocator,
														&mLineRendererAllocator,
														&mRigidbodyAllocator,
														&mCameraAllocator,
														&mLightAllocator,
														&mSphereColliderAllocator,
														&mBoxColliderAllocator,
														&mCapsuleColliderAllocator,
														&mMeshColliderAllocator, 
														componentType, 
														index);
		}
		else{
			swappedComponent = destroyComponent(&mComponentAllocatorMap, componentType, index);
		}

		mIdToGlobalIndex.erase(it2);

		if(swappedComponent != NULL){
			mComponentIdsMarkedMoved.push_back(make_triple(swappedComponent->mComponentId, componentType, mIdToGlobalIndex[swappedComponent->mComponentId]));

			mIdToGlobalIndex[swappedComponent->mComponentId] = index;
		}
	}
	else{
		std::string message = "Error: component id " + componentId.toString() + " not found in map when trying to destroy\n";
		Log::error(message.c_str());
	} 

	//remove from id to type map
	std::unordered_map<Guid, int>::iterator it3 = mIdToType.find(componentId);
	if (it3 != mIdToType.end()) {
		mIdToType.erase(it3);
	}
	else {
		std::string message = "Error: Could not find component with id " + componentId.toString() + " when trying to delete from id to type map\n";
		Log::error(message.c_str());
		return;
	}
}

bool World::isMarkedForLatentDestroy(const Guid& id)
{
	for(size_t i = 0; i < mEntityIdsMarkedLatentDestroy.size(); i++){
		if(mEntityIdsMarkedLatentDestroy[i] == id){
			return true;
		}
	}

	for(size_t i = 0; i < mComponentIdsMarkedLatentDestroy.size(); i++){
		if(mComponentIdsMarkedLatentDestroy[i].second == id){
			return true;
		}
	}

	return false;
}

void World::clearIdsMarkedCreatedOrDestroyed()
{
	mEntityIdsMarkedCreated.clear();
	mEntityIdsMarkedLatentDestroy.clear();
	mComponentIdsMarkedCreated.clear();
	mComponentIdsMarkedLatentDestroy.clear();
}

void World::clearIdsMarkedMoved()
{
	mEntityIdsMarkedMoved.clear();
	mComponentIdsMarkedMoved.clear();
}

std::vector<std::pair<Guid, int>> World::getComponentsOnEntity(const Guid& entityId)
{
	std::vector<std::pair<Guid, int>> componentsOnEntity;

	std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::iterator it = mEntityIdToComponentIds.find(entityId);
	if(it != mEntityIdToComponentIds.end()){
		componentsOnEntity = it->second;
	}

	return componentsOnEntity;
}

std::vector<Guid> World::getEntityIdsMarkedCreated() const
{
	return mEntityIdsMarkedCreated;
}

std::vector<Guid> World::getEntityIdsMarkedLatentDestroy() const
{
	return mEntityIdsMarkedLatentDestroy;
}

std::vector<std::pair<Guid, int>> World::getEntityIdsMarkedMoved() const
{
	return mEntityIdsMarkedMoved;
}
			
std::vector<triple<Guid, Guid, int>> World::getComponentIdsMarkedCreated() const
{
	return mComponentIdsMarkedCreated;
}

std::vector<triple<Guid, Guid, int>> World::getComponentIdsMarkedLatentDestroy() const
{
	return mComponentIdsMarkedLatentDestroy;
}

std::vector<triple<Guid, int, int>> World::getComponentIdsMarkedMoved() const
{
	return mComponentIdsMarkedMoved;
}

std::string World::getAssetFilepath(const Guid& assetId) const
{
	std::unordered_map<Guid, std::string>::const_iterator it = mAssetIdToFilepath.find(assetId);
	if (it != mAssetIdToFilepath.end()) {
		return it->second;
	}

	return "";
}

std::string World::getSceneFilepath(const Guid& sceneId) const
{
	std::unordered_map<Guid, std::string>::const_iterator it = mSceneIdToFilepath.find(sceneId);
	if (it != mSceneIdToFilepath.end()) {
		return it->second;
	}

	return "";
}

Guid World::getSphereMesh() const 
{
	return mSphereMeshId;
}

Guid World::getCubeMesh() const
{
	return mCubeMeshId;
}

Guid World::getColorMaterial() const
{
	return mColorMaterialId;
}

Guid World::getSimpleLitMaterial() const
{
	return mSimpleLitMaterialId;
}
//bool World::raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance)
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
//bool World::raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance, Collider** collider)
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