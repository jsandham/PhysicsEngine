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
	WorldAllocators allocators;
	allocators.mMeshAllocator = &mMeshAllocator;
	allocators.mMaterialAllocator = &mMaterialAllocator;
	allocators.mShaderAllocator = &mShaderAllocator;
	allocators.mTexture2DAllocator = &mTexture2DAllocator;
	allocators.mTexture3DAllocator = &mTexture3DAllocator;
	allocators.mCubemapAllocator = &mCubemapAllocator;
	allocators.mFontAllocator = &mFontAllocator;
	allocators.mAssetAllocatorMap = &mAssetAllocatorMap;

	WorldIsState idState;
	idState.mMeshIdToGlobalIndex = &mMeshIdToGlobalIndex;
	idState.mMaterialIdToGlobalIndex = &mMaterialIdToGlobalIndex;
	idState.mShaderIdToGlobalIndex = &mShaderIdToGlobalIndex;
	idState.mTexture2DIdToGlobalIndex = &mTexture2DIdToGlobalIndex;
	idState.mTexture3DIdToGlobalIndex = &mTexture3DIdToGlobalIndex;
	idState.mCubemapIdToGlobalIndex = &mCubemapIdToGlobalIndex;
	idState.mFontIdToGlobalIndex = &mFontIdToGlobalIndex;
	idState.mIdToGlobalIndex = &mIdToGlobalIndex;
	idState.mIdToType = &mIdToType;

	PhysicsEngine::loadAssetIntoWorld(filePath, 
									  allocators,
									  idState,
									  mAssetIdToFilepath);

	return true;
}

bool World::loadScene(const std::string& filePath, bool ignoreSystemsAndCamera)
{
	WorldAllocators allocators;
	allocators.mEntityAllocator = &mEntityAllocator;
	allocators.mTransformAllocator = &mTransformAllocator;
	allocators.mMeshRendererAllocator = &mMeshRendererAllocator;
	allocators.mLineRendererAllocator = &mLineRendererAllocator;
	allocators.mRigidbodyAllocator = &mRigidbodyAllocator;
	allocators.mCameraAllocator = &mCameraAllocator;
	allocators.mLightAllocator = &mLightAllocator;
	allocators.mSphereColliderAllocator = &mSphereColliderAllocator;
	allocators.mBoxColliderAllocator = &mBoxColliderAllocator;
	allocators.mCapsuleColliderAllocator = &mCapsuleColliderAllocator;
	allocators.mMeshColliderAllocator = &mMeshColliderAllocator;
	allocators.mRenderSystemAllocator = &mRenderSystemAllocator;
	allocators.mPhysicsSystemAllocator = &mPhysicsSystemAllocator;
	allocators.mCleanupSystemAllocator = &mCleanupSystemAllocator;
	allocators.mDebugSystemAllocator = &mDebugSystemAllocator;
	allocators.mComponentAllocatorMap = &mComponentAllocatorMap;
	allocators.mSystemAllocatorMap = &mSystemAllocatorMap;

	WorldIsState idState;
	idState.mEntityIdToGlobalIndex = &mEntityIdToGlobalIndex;
	idState.mTransformIdToGlobalIndex = &mTransformIdToGlobalIndex;
	idState.mMeshRendererIdToGlobalIndex = &mMeshRendererIdToGlobalIndex;
	idState.mLineRendererIdToGlobalIndex = &mLineRendererIdToGlobalIndex;
	idState.mRigidbodyIdToGlobalIndex = &mRigidbodyIdToGlobalIndex;
	idState.mCameraIdToGlobalIndex = &mCameraIdToGlobalIndex;
	idState.mLightIdToGlobalIndex = &mLightIdToGlobalIndex;
	idState.mSphereColliderIdToGlobalIndex = &mSphereColliderIdToGlobalIndex;
	idState.mBoxColliderIdToGlobalIndex = &mBoxColliderIdToGlobalIndex;
	idState.mCapsuleColliderIdToGlobalIndex = &mCapsuleColliderIdToGlobalIndex;
	idState.mMeshColliderIdToGlobalIndex = &mMeshColliderIdToGlobalIndex;
	idState.mRenderSystemIdToGlobalIndex = &mRenderSystemIdToGlobalIndex;
	idState.mPhysicsSystemIdToGlobalIndex = &mPhysicsSystemIdToGlobalIndex;
	idState.mCleanupSystemIdToGlobalIndex = &mCleanupSystemIdToGlobalIndex;
	idState.mDebugSystemIdToGlobalIndex = &mDebugSystemIdToGlobalIndex;
	idState.mIdToGlobalIndex = &mIdToGlobalIndex;
	idState.mIdToType = &mIdToType;
	idState.mEntityIdToComponentIds = &mEntityIdToComponentIds;
	idState.mEntityIdsMarkedCreated = &mEntityIdsMarkedCreated;
	idState.mComponentIdsMarkedCreated = &mComponentIdsMarkedCreated;

	PhysicsEngine::loadSceneIntoWorld(filePath,
									  allocators,
									  idState,
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

Entity* World::getEntityById(const Guid& entityId)
{
	return getById_impl<Entity>(mEntityIdToGlobalIndex, &mEntityAllocator, entityId);
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

		addIdToGlobalIndexMap_impl<Entity>(entity->mEntityId, globalIndex, type);

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
		addIdToGlobalIndexMap_impl<Entity>(entity->mEntityId, globalIndex, type);

		mEntityIdToComponentIds[entity->mEntityId] = std::vector<std::pair<Guid, int>>();

		mEntityIdsMarkedCreated.push_back(entity->mEntityId);
	}

	return entity;
}

void World::latentDestroyEntity(const Guid& entityId)
{
	mEntityIdsMarkedLatentDestroy.push_back(entityId);

	// add any components found on the entity to the latent destroy component list
	std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::const_iterator it = mEntityIdToComponentIds.find(entityId);

	assert(it != mEntityIdToComponentIds.end());

	for(size_t i = 0; i < it->second.size(); i++){
		latentDestroyComponent(entityId, it->second[i].first, it->second[i].second);
	}
}

void World::immediateDestroyEntity(const Guid& entityId)
{
	std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::const_iterator it = mEntityIdToComponentIds.find(entityId);

	assert(it != mEntityIdToComponentIds.end());

	for(size_t i = 0; i < it->second.size(); i++){
		immediateDestroyComponent(entityId, it->second[i].first, it->second[i].second);
	}

	mEntityIdToComponentIds.erase(it);

	int index = getIndexOf(entityId);

	assert(index != -1);

	Entity* swappedEntity = destroyInternalEntity(&mEntityAllocator, index);

	removeInternalEntityIdFromIndexMap(&mEntityIdToGlobalIndex,
									   &mIdToGlobalIndex,
									   &mIdToType,
									   entityId);

	if (swappedEntity != NULL) {
		addInternalEntityIdToIndexMap(&mEntityIdToGlobalIndex,
			&mIdToGlobalIndex,
			&mIdToType,
			swappedEntity->mEntityId,
			index);
	}
}

void World::latentDestroyComponent(const Guid& entityId, const Guid& componentId, int componentType)
{
	mComponentIdsMarkedLatentDestroy.push_back(make_triple(entityId, componentId, componentType));
}

void World::immediateDestroyComponent(const Guid& entityId, const Guid &componentId, int componentType)
{
	std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::iterator it = mEntityIdToComponentIds.find(entityId);

	assert(it != mEntityIdToComponentIds.end());

	for(size_t i = 0; i < it->second.size(); i++){
		if(it->second[i].first == componentId){
			it->second.erase(it->second.begin() + i);
			break;
		}
	}

	int index = getIndexOf(componentId);

	assert(index != -1);

	Component* swappedComponent = NULL;
	if (Component::isInternal(componentType)) {
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
	else {
		swappedComponent = destroyComponent(&mComponentAllocatorMap, componentType, index);
	}

	if (Component::isInternal(componentType)) {
		removeInternalComponentIdFromIndexMap(&mTransformIdToGlobalIndex,
											  &mMeshRendererIdToGlobalIndex,
											  &mLineRendererIdToGlobalIndex,
											  &mRigidbodyIdToGlobalIndex,
											  &mCameraIdToGlobalIndex,
											  &mLightIdToGlobalIndex,
											  &mSphereColliderIdToGlobalIndex,
											  &mBoxColliderIdToGlobalIndex,
										 	  &mCapsuleColliderIdToGlobalIndex,
										 	  &mMeshColliderIdToGlobalIndex,
											  &mIdToGlobalIndex,
										 	  &mIdToType,
											  componentId,
											  componentType);

		if (swappedComponent != NULL) {
			addInternalComponentIdToIndexMap(&mTransformIdToGlobalIndex,
				&mMeshRendererIdToGlobalIndex,
				&mLineRendererIdToGlobalIndex,
				&mRigidbodyIdToGlobalIndex,
				&mCameraIdToGlobalIndex,
				&mLightIdToGlobalIndex,
				&mSphereColliderIdToGlobalIndex,
				&mBoxColliderIdToGlobalIndex,
				&mCapsuleColliderIdToGlobalIndex,
				&mMeshColliderIdToGlobalIndex,
				&mIdToGlobalIndex,
				&mIdToType,
				swappedComponent->mComponentId,
				componentType,
				index);
		}
	}
	else {
		removeComponentIdFromIndexMap(&mIdToGlobalIndex, &mIdToType, componentId, componentType);

		if (swappedComponent != NULL) {
			addComponentIdToIndexMap(&mIdToGlobalIndex, &mIdToType, swappedComponent->mComponentId, componentType, index);
		}
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

std::vector<std::pair<Guid, int>> World::getComponentsOnEntity(const Guid& entityId)
{
	std::vector<std::pair<Guid, int>> componentsOnEntity;

	std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::const_iterator it = mEntityIdToComponentIds.find(entityId);
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
			
std::vector<triple<Guid, Guid, int>> World::getComponentIdsMarkedCreated() const
{
	return mComponentIdsMarkedCreated;
}

std::vector<triple<Guid, Guid, int>> World::getComponentIdsMarkedLatentDestroy() const
{
	return mComponentIdsMarkedLatentDestroy;
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

Guid World::getFontShaderId() const
{
	return mFontShaderId;
}

Guid World::getColorShaderId() const
{
	return mColorShaderId;
}

Guid World::getPositionAndNormalsShaderId() const
{
	return mPositionAndNormalsShaderId;
}

Guid World::getSsaoShaderId() const
{
	return mSsaoShaderId;
}

Guid World::getScreenQuadShaderId() const
{
	return mScreenQuadShaderId;
}

Guid World::getNormalMapShaderId() const
{
	return mNormalMapShaderId;
}

Guid World::getDepthMapShaderId() const
{
	return mDepthMapShaderId;
}

Guid World::getShadowDepthMapShaderId() const
{
	return mShadowDepthMapShaderId;
}

Guid World::getShadowDepthCubemapShaderId() const
{
	return mShadowDepthCubemapShaderId;
}

Guid World::getGbufferShaderId() const
{
	return mGbufferShaderId;
}

Guid World::getSimpleLitShaderId() const
{
	return mSimpleLitShaderId;
}

Guid World::getSimpleLitDeferredShaderId() const
{
	return mSimpleLitDeferedShaderId;
}

Guid World::getOverdrawShaderId() const
{
	return mOverdrawShaderId;
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