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
	std::unordered_map<int, Allocator*>::iterator it1 = mComponentAllocatorMap.begin();
	for (it1 == mComponentAllocatorMap.begin(); it1 != mComponentAllocatorMap.end(); it1++) {
		delete it1->second;
	}

	std::unordered_map<int, Allocator*>::iterator it2 = mSystemAllocatorMap.begin();
	for (it2 == mSystemAllocatorMap.begin(); it2 != mSystemAllocatorMap.end(); it2++) {
		delete it2->second;
	}

	std::unordered_map<int, Allocator*>::iterator it3 = mAssetAllocatorMap.begin();
	for (it3 == mAssetAllocatorMap.begin(); it3 != mAssetAllocatorMap.end(); it3++) {
		delete it3->second;
	}
}

bool World::loadAsset(const std::string& filePath)
{
	Log::info(("file path: " + filePath + "\n").c_str());

	std::ifstream file;
	file.open(filePath, std::ios::binary);

	if(!file.is_open()){
		std::string errorMessage = "Failed to open asset bundle " + filePath + "\n";
		Log::error(&errorMessage[0]);
		return false;
	}
	
	AssetHeader header;
	file.read(reinterpret_cast<char*>(&header), sizeof(AssetHeader));

	while( file.peek() != EOF )
	{
		char classification;
		int type;
		size_t size;

		file.read(reinterpret_cast<char*>(&classification), sizeof(char));
		file.read(reinterpret_cast<char*>(&type), sizeof(int));
		file.read(reinterpret_cast<char*>(&size), sizeof(size_t));

		if(type <= -1){
			Log::error("Type cannot be less than 0 when reading asset bundle file\n");
			return false;
		}

		if(size <= 0){
			Log::error("Size cannot be less than 1 when reading asset bundle file\n");
			return false;
		}

		std::vector<char> data(size);
		file.read(reinterpret_cast<char*>(&data[0]), data.size() * sizeof(char));		

		int index = -1;
		Asset* asset = NULL;
		if(type < 20){
			asset = PhysicsEngine::loadInternalAsset(&mMeshAllocator,
													 &mMaterialAllocator,
													 &mShaderAllocator,
													 &mTexture2DAllocator,
													 &mTexture3DAllocator,
													 &mCubemapAllocator,
													 &mFontAllocator, 
													 data, 
													 type, 
													 &index);
		}
		else{
			asset = PhysicsEngine::loadAsset(&mAssetAllocatorMap, data, type, &index);
		}

		if(asset == NULL || index == -1){
			Log::error("Error: Could not load asset\n");
			return false;
		}

		assetIdToFilepath[asset->mAssetId] = filePath;

		if(mIdToGlobalIndex.find(asset->mAssetId) == mIdToGlobalIndex.end()){
			mIdToGlobalIndex[asset->mAssetId] = index;
		}
		else{
			std::string errorMessage = "Asset with id " + asset->mAssetId.toString() + " already exists in id to global index map\n";
			Log::error(&errorMessage[0]);
			return false;
		}

		if (mIdToType.find(asset->mAssetId) == mIdToType.end()){
			mIdToType[asset->mAssetId] = type;
		}
		else {
			std::string errorMessage = "Asset with id " + asset->mAssetId.toString() + " already exists in id to type map\n";
			Log::error(&errorMessage[0]);
			return false;
		}
	}

	file.close();

	return true;
}

bool World::loadScene(const std::string& filePath, bool ignoreSystemsAndCamera)
{
	std::ifstream file;
	file.open(filePath, std::ios::binary);

	if(!file.is_open()){
		std::string errorMessage = "Failed to open scene file " + filePath + "\n";
		Log::error(&errorMessage[0]);
		return false;
	}

	SceneHeader sceneHeader;
	file.read(reinterpret_cast<char*>(&sceneHeader), sizeof(SceneHeader));
	while( file.peek() != EOF ){
		char classification;
		int type;
		size_t size;

		file.read(reinterpret_cast<char*>(&classification), sizeof(char));
		file.read(reinterpret_cast<char*>(&type), sizeof(int));
		file.read(reinterpret_cast<char*>(&size), sizeof(size_t));

		if(type <= -1){
			Log::error("Type cannot be less than 0 when reading scene file\n");
			return false;
		}

		if(size <= 0){
			Log::error("Size cannot be less than 1 when reading scene file\n");
			return false;
		}

		std::vector<char> data(size);

		file.read(reinterpret_cast<char*>(&data[0]), data.size() * sizeof(char));

		int index = -1;
		if(classification == 'e'){
			Entity* entity = NULL;
			if(type == 0){
				entity = PhysicsEngine::loadInternalEntity(&mEntityAllocator, data, &index);
			}
			else{
				Log::error("Entity must be of type 0\n");
				return false;
			}

			if(entity == NULL || index == -1){
				std::string errorMessage = "Could not load entity corresponding to type " + std::to_string(type) + "\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			if(mIdToGlobalIndex.find(entity->mEntityId) == mIdToGlobalIndex.end()){
				mIdToGlobalIndex[entity->mEntityId] = index;

				mEntityIdsMarkedCreated.push_back(entity->mEntityId);
			}
			else{
				std::string errorMessage = "Entity with id " + entity->mEntityId.toString() + " already exists in id to global index map\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			if (mIdToType.find(entity->mEntityId) == mIdToType.end()) {
				mIdToType[entity->mEntityId] = type;
			}
			else {
				std::string errorMessage = "Entity with id " + entity->mEntityId.toString() + " already exists in id to type map\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::iterator it = mEntityIdToComponentIds.find(entity->mEntityId);
			if (it == mEntityIdToComponentIds.end()) {
				mEntityIdToComponentIds[entity->mEntityId] = std::vector<std::pair<Guid, int>>();
			}
		}
		else if(classification == 'c'){
			if (ignoreSystemsAndCamera && type == ComponentType<Camera>::type) {
				continue;
			}

			Component* component = NULL;
			if(type < 20){
				component = PhysicsEngine::loadInternalComponent(&mTransformAllocator,
																 &mMeshRendererAllocator,
																 &mLineRendererAllocator,
																 &mRigidbodyAllocator,
																 &mCameraAllocator,
																 &mLightAllocator,
																 &mSphereColliderAllocator,
																 &mBoxColliderAllocator,
																 &mCapsuleColliderAllocator,
																 &mMeshColliderAllocator, 
																 data, 
																 type, 
																 &index);
			}
			else{
				component = PhysicsEngine::loadComponent(&mComponentAllocatorMap, data, type, &index);
			}

			if(component == NULL || index == -1){
				std::string errorMessage = "Could not load component corresponding to type " + std::to_string(type) + "\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			if(mIdToGlobalIndex.find(component->mComponentId) == mIdToGlobalIndex.end()){
				mIdToGlobalIndex[component->mComponentId] = index;

				mComponentIdsMarkedCreated.push_back(make_triple(component->mEntityId, component->mComponentId, type));
			}
			else{
				std::string errorMessage = "Component with id " + component->mComponentId.toString() + " already exists in id to global index map\n";
				Log::error(&errorMessage[0]);
				return false;
			}
			
			if (mIdToType.find(component->mComponentId) == mIdToType.end()) {
				mIdToType[component->mComponentId] = type;
			}
			else {
				std::string errorMessage = "Component with id " + component->mComponentId.toString() + " already exists in id to type map\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			mEntityIdToComponentIds[component->mEntityId].push_back(std::make_pair(component->mComponentId, type));
		}
		else if(classification == 's' && !ignoreSystemsAndCamera){
			System* system = NULL;
			if(type < 20){
				system = PhysicsEngine::loadInternalSystem(&mRenderSystemAllocator,
														   &mPhysicsSystemAllocator,
														   &mCleanupSystemAllocator,
														   &mDebugSystemAllocator, 
														   data, 
														   type, 
														   &index);
			}
			else{
				system = PhysicsEngine::loadSystem(&mSystemAllocatorMap, data, type, &index);
			}

			if(system == NULL || index == -1){
				std::string errorMessage = "Could not load system corresponding to type " + std::to_string(type) + "\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			if (mIdToGlobalIndex.find(system->mSystemId) == mIdToGlobalIndex.end()) {
				mIdToGlobalIndex[system->mSystemId] = index;
			}
			else {
				std::string errorMessage = "System with id " + system->mSystemId.toString() + " already exists in id to global index map\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			if (mIdToType.find(system->mSystemId) == mIdToType.end()) {
				mIdToType[system->mSystemId] = type;
			}
			else {
				std::string errorMessage = "System with id " + system->mSystemId.toString() + " already exists in id to type map\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			// maybe set system in vector??
			mSystems.push_back(system);
		}
		else{
			Log::error("Classification must be \'e\' (entity), \'c\' (component), or \'s\' (system)");
			return false;
		}
	}

	// sort systems by order
	if(!ignoreSystemsAndCamera){
		for(size_t i = 0; i < mSystems.size(); i++){
			int minOrder = mSystems[i]->getOrder();
			int minOrderIndex = (int)i;
			for(size_t j = i + 1; j < mSystems.size(); j++){
				if(mSystems[j]->getOrder() < minOrder){
					minOrder = mSystems[j]->getOrder();
					minOrderIndex = (int)j;
				}
			}

			System* temp = mSystems[i];
			mSystems[i] = mSystems[minOrderIndex];
			mSystems[minOrderIndex] = temp;
		}
	}

	file.close();

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

int World::getNumberOfSystems() const
{
	return (int)mSystems.size();
}

// error here if entityId does not correspond to an entity but instead an asset or component
Entity* World::getEntity(const Guid& entityId)
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

System* World::getSystemByIndex(int index)
{
	if (index < 0 || index >= mSystems.size()) {
		return NULL;
	}

	return mSystems[index];
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

void World::latentDestroyEntity(Guid entityId)
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

void World::immediateDestroyEntity(Guid entityId)
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

void World::latentDestroyComponent(Guid entityId, Guid componentId, int componentType)
{
	mComponentIdsMarkedLatentDestroy.push_back(make_triple(entityId, componentId, componentType));
}

void World::immediateDestroyComponent(Guid entityId, Guid componentId, int componentType)
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

bool World::isMarkedForLatentDestroy(Guid id)
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

std::vector<std::pair<Guid, int>> World::getComponentsOnEntity(Guid entityId)
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
	std::unordered_map<Guid, std::string>::const_iterator it = assetIdToFilepath.find(assetId);
	if (it != assetIdToFilepath.end()) {
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