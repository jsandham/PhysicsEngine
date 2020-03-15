#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_set>

#include "../../include/core/Log.h"
#include "../../include/core/Load.h"
#include "../../include/core/LoadInternal.h"
#include "../../include/core/World.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

World::World()
{
	entityAllocator = NULL;
	bounds.centre = glm::vec3(0.0f, 0.0f, 40.0f);
	bounds.size = 2.0f * glm::vec3(200.0f, 200.0f, 200.0f);

	//stree.create(bounds, 2, 5);
	//dtree.create(bounds, 2, 5);

	debug = false;
	debugView = -1;
}

World::~World()
{
	if (entityAllocator != NULL) {
		delete entityAllocator;
	}

	std::map<int, Allocator*>::iterator it1 = componentAllocatorMap.begin();
	for (it1 == componentAllocatorMap.begin(); it1 != componentAllocatorMap.end(); it1++) {
		delete it1->second;
	}

	std::map<int, Allocator*>::iterator it2 = systemAllocatorMap.begin();
	for (it2 == systemAllocatorMap.begin(); it2 != systemAllocatorMap.end(); it2++) {
		delete it2->second;
	}

	std::map<int, Allocator*>::iterator it3 = assetAllocatorMap.begin();
	for (it3 == assetAllocatorMap.begin(); it3 != assetAllocatorMap.end(); it3++) {
		delete it3->second;
	}
}

bool World::loadAsset(std::string filePath)
{
	std::string message = "Attempting to load asset " + filePath + " into world\n";
	Log::info(&message[0]);

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
			asset = PhysicsEngine::loadInternalAsset(&assetAllocatorMap, data, type, &index);
		}
		else{
			asset = PhysicsEngine::loadAsset(&assetAllocatorMap, data, type, &index);
		}

		if(asset == NULL || index == -1){
			Log::error("Error: Could not load asset");
			return false;
		}

		if(idToGlobalIndex.find(asset->assetId) == idToGlobalIndex.end()){
			idToGlobalIndex[asset->assetId] = index;
		}
		else{
			std::string errorMessage = "Asset with id " + asset->assetId.toString() + " already exists in id to global index map\n";
			Log::error(&errorMessage[0]);
			return false;
		}

		if (idToType.find(asset->assetId) == idToType.end()){
			idToType[asset->assetId] = type;
		}
		else {
			std::string errorMessage = "Asset with id " + asset->assetId.toString() + " already exists in id to type map\n";
			Log::error(&errorMessage[0]);
			return false;
		}
	}

	file.close();

	return true;
}

bool World::loadScene(std::string filePath, bool ignoreSystemsAndCamera)
{
	Log::info("Attempting to load scene\n");

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
				entity = PhysicsEngine::loadInternalEntity(entityAllocator, data, &index);
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

			if(idToGlobalIndex.find(entity->entityId) == idToGlobalIndex.end()){
				idToGlobalIndex[entity->entityId] = index;

				entityIdsMarkedCreated.push_back(entity->entityId);
			}
			else{
				std::string errorMessage = "Entity with id " + entity->entityId.toString() + " already exists in id to global index map\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			if (idToType.find(entity->entityId) == idToType.end()) {
				idToType[entity->entityId] = type;
			}
			else {
				std::string errorMessage = "Entity with id " + entity->entityId.toString() + " already exists in id to type map\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			std::map<Guid, std::vector<std::pair<Guid, int>>>::iterator it = entityIdToComponentIds.find(entity->entityId);
			if (it == entityIdToComponentIds.end()) {
				entityIdToComponentIds[entity->entityId] = std::vector<std::pair<Guid, int>>();
			}
		}
		else if(classification == 'c'){
			if (ignoreSystemsAndCamera && type == ComponentType<Camera>::type) {
				continue;
			}

			Component* component = NULL;
			if(type < 20){
				component = PhysicsEngine::loadInternalComponent(&componentAllocatorMap, data, type, &index);
			}
			else{
				component = PhysicsEngine::loadComponent(&componentAllocatorMap, data, type, &index);
			}

			if(component == NULL || index == -1){
				std::string errorMessage = "Could not load component corresponding to type " + std::to_string(type) + "\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			if(idToGlobalIndex.find(component->componentId) == idToGlobalIndex.end()){
				idToGlobalIndex[component->componentId] = index;

				componentIdsMarkedCreated.push_back(make_triple(component->entityId, component->componentId, type));
			}
			else{
				std::string errorMessage = "Component with id " + component->componentId.toString() + " already exists in id to global index map\n";
				Log::error(&errorMessage[0]);
				return false;
			}
			
			if (idToType.find(component->componentId) == idToType.end()) {
				idToType[component->componentId] = type;
			}
			else {
				std::string errorMessage = "Component with id " + component->componentId.toString() + " already exists in id to type map\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			entityIdToComponentIds[component->entityId].push_back(std::make_pair(component->componentId, type));
		}
		else if(classification == 's' && !ignoreSystemsAndCamera){
			System* system = NULL;
			if(type < 20){
				system = PhysicsEngine::loadInternalSystem(&systemAllocatorMap, data, type, &index);
			}
			else{
				system = PhysicsEngine::loadSystem(&systemAllocatorMap, data, type, &index);
			}

			if(system == NULL || index == -1){
				std::string errorMessage = "Could not load system corresponding to type " + std::to_string(type) + "\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			if (idToGlobalIndex.find(system->systemId) == idToGlobalIndex.end()) {
				idToGlobalIndex[system->systemId] = index;
			}
			else {
				std::string errorMessage = "System with id " + system->systemId.toString() + " already exists in id to global index map\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			if (idToType.find(system->systemId) == idToType.end()) {
				idToType[system->systemId] = type;
			}
			else {
				std::string errorMessage = "System with id " + system->systemId.toString() + " already exists in id to type map\n";
				Log::error(&errorMessage[0]);
				return false;
			}

			// maybe set system in vector??
			systems.push_back(system);
		}
		else{
			Log::error("Classification must be \'e\' (entity), \'c\' (component), or \'s\' (system)");
			return false;
		}
	}

	// sort systems by order
	if(!ignoreSystemsAndCamera){
		for(size_t i = 0; i < systems.size(); i++){
			int minOrder = systems[i]->getOrder();
			int minOrderIndex = (int)i;
			for(size_t j = i + 1; j < systems.size(); j++){
				if(systems[j]->getOrder() < minOrder){
					minOrder = systems[j]->getOrder();
					minOrderIndex = (int)j;
				}
			}

			System* temp = systems[i];
			systems[i] = systems[minOrderIndex];
			systems[minOrderIndex] = temp;
		}
	}

	file.close();

	return true;
}

bool World::loadSceneFromEditor(std::string filePath)
{
	return loadScene(filePath, true);
}


void World::latentDestroyEntitiesInWorld() // clearLatent? latentDestroyEntitiesInWorld?
{
	// latent destroy all entities (and thereby also all components)
	for (int i = 0; i < getNumberOfEntities(); i++) {
		Entity* entity = getEntityByIndex(i);

		if (!entity->doNotDestroy) {
			std::string message = "Adding entity: " + entity->entityId.toString() + " to latent destroy list\n";
			Log::warn(message.c_str());
			latentDestroyEntity(entity->entityId);
		}
		else {
			std::string message = "Warn: Skipping entity: " + entity->entityId.toString() + " as it is marked do not destroy\n";
			Log::warn(message.c_str());
		}
	}
}

int World::getNumberOfEntities()
{
	PoolAllocator<Entity>* allocator = getEntityAllocator();
	if (allocator == NULL) {
		return 0;
	}

	return (int)allocator->getCount();
}

int World::getNumberOfSystems()
{
	return (int)systems.size();
}

Entity* World::getEntity(Guid entityId)
{
	assert(entityId != Guid::INVALID);

	PoolAllocator<Entity>* allocator = getEntityAllocator();
	if (allocator == NULL) {
		return NULL;
	}

	std::map<Guid, int>::iterator it = idToGlobalIndex.find(entityId);
	if(it != idToGlobalIndex.end()){
		return allocator->get(it->second);
	}
	else{
		std::string message = "Error: No entity with id " + entityId.toString() + " was found\n";
		Log::error(message.c_str());
		return NULL;
	}
}

Entity* World::getEntityByIndex(int index)
{
	PoolAllocator<Entity>* allocator = getEntityAllocator();
	if (allocator == NULL) {
		return NULL;
	}

	return allocator->get(index);
}

System* World::getSystemByIndex(int index)
{
	return systems[index];
}

int World::getIndexOf(Guid id)
{
	std::map<Guid, int>::iterator it = idToGlobalIndex.find(id);
	if( it != idToGlobalIndex.end()){
		return it->second;
	}

	return -1;
}

int World::getTypeOf(Guid id)
{
	std::map<Guid, int>::iterator it = idToType.find(id);
	if (it != idToType.end()) {
		return it->second;
	}

	return -1;
}

//int World::getIndexOfAsset(Guid id)
//{
//	std::map<Guid, int>::iterator it2 = idToGlobalIndex.find(id);
//	if( it2 != idToGlobalIndex.end()){
//		return it2->second;
//	}
//
//	return -1;
//}

Entity* World::createEntity()
{
	PoolAllocator<Entity>* allocator = getEntityOrAddAllocator();

	int globalIndex = (int)allocator->getCount();
	int type = EntityType<Entity>::type;
	Guid entityId = Guid::newGuid();

	Entity* entity = allocator->construct();
	entity->entityId = entityId;
	entity->doNotDestroy = false;

	idToGlobalIndex[entityId] = globalIndex;
	idToType[entityId] = type;
	
	entityIdToComponentIds[entityId] = std::vector<std::pair<Guid, int>>();

	entityIdsMarkedCreated.push_back(entityId);

	return entity;
}

Entity* World::createEntity(Guid entityId)
{
	Entity* oldEntity = getEntity(entityId);

	if(oldEntity == NULL){ 
		std::cout << "Error: Could not find entity (" << entityId.toString() << ") when trying to create entity" << std::endl;
		return NULL;
	}

	std::vector<std::pair<Guid, int>> oldComponents;

	std::map<Guid, std::vector<std::pair<Guid, int>>>::iterator it = entityIdToComponentIds.find(entityId);
	if(it != entityIdToComponentIds.end()){
		oldComponents = it->second;
	}
	else{
		std::cout << "Error: " << std::endl;
		return NULL;
	}

	PoolAllocator<Entity>* allocator = getEntityOrAddAllocator();
	if (allocator == NULL) {
		return NULL;
	}

	Entity* newEntity = allocator->construct();

	// add components to new entity
	for(size_t i = 0; i < oldComponents.size(); i++){
		Guid oldComponentId = oldComponents[i].first;
		int oldComponentType = oldComponents[i].second;

		// TODO: How do I add the new component using the instance type and old component Id???
	}

	return NULL;
}

Entity* World::createEntity(std::vector<char> data)
{
	PoolAllocator<Entity>* allocator = getEntityOrAddAllocator();

	int globalIndex = (int)allocator->getCount();
	int type = EntityType<Entity>::type;

	Entity* entity = allocator->construct(data);

	idToGlobalIndex[entity->entityId] = globalIndex;
	idToType[entity->entityId] = type;

	entityIdToComponentIds[entity->entityId] = std::vector<std::pair<Guid, int>>();

	entityIdsMarkedCreated.push_back(entity->entityId);

	return entity;
}

Camera* World::createEditorCamera()
{
	PoolAllocator<Entity>* allocator1 = getEntityOrAddAllocator();
	PoolAllocator<Transform>* allocator2 = getComponentOrAddAllocator<Transform>();
	PoolAllocator<Camera>* allocator3 = getComponentOrAddAllocator<Camera>();

	// Editor entity
	int globalIndex = (int)allocator1->getCount();
	int type = EntityType<Entity>::type;
	Guid entityId = Guid("11111111-1111-1111-1111-111111111111");
	Entity* entity = allocator1->construct();
	entity->entityId = entityId;
	entity->doNotDestroy = true;

	idToGlobalIndex[entityId] = globalIndex;
	idToType[entityId] = type;
	entityIdToComponentIds[entityId] = std::vector<std::pair<Guid, int>>();
	entityIdsMarkedCreated.push_back(entityId);

	// editor only transform
	int transformGlobalIndex = (int)allocator2->getCount();
	int transformType = ComponentType<Transform>::type;
	Guid transformId = Guid("22222222-2222-2222-2222-222222222222");
	Transform* transform = allocator2->construct();
	transform->entityId = entityId;
	transform->componentId = transformId;

	idToGlobalIndex[transformId] = transformGlobalIndex;
	idToType[transformId] = transformType;
	entityIdToComponentIds[entityId].push_back(std::make_pair(transformId, transformType));
	componentIdsMarkedCreated.push_back(make_triple(entityId, transformId, transformType));

	// editor only camera
	int cameraGlobalIndex = (int)allocator3->getCount();
	int cameraType = ComponentType<Camera>::type;
	Guid cameraId = Guid("33333333-3333-3333-3333-333333333333");
	Camera* camera = allocator3->construct();
	camera->entityId = entityId;
	camera->componentId = cameraId;

	idToGlobalIndex[cameraId] = cameraGlobalIndex;
	idToType[cameraId] = cameraType;
	entityIdToComponentIds[entityId].push_back(std::make_pair(cameraId, cameraType));
	componentIdsMarkedCreated.push_back(make_triple(entityId, cameraId, cameraType));

	return camera;
}

void World::latentDestroyEntity(Guid entityId)
{
	entityIdsMarkedLatentDestroy.push_back(entityId);

	// add any components found on the entity to the latent destroy component list
	std::map<Guid, std::vector<std::pair<Guid, int>>>::iterator it = entityIdToComponentIds.find(entityId);
	if(it != entityIdToComponentIds.end()){
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
	std::map<Guid, std::vector<std::pair<Guid, int>>>::iterator it1 = entityIdToComponentIds.find(entityId);
	if(it1 != entityIdToComponentIds.end()){
		std::vector<std::pair<Guid, int>> componentsOnEntity = it1->second;

		for(size_t i = 0; i < componentsOnEntity.size(); i++){
			Guid componentId = componentsOnEntity[i].first;
			int componentType = componentsOnEntity[i].second;

			immediateDestroyComponent(entityId, componentId, componentType);
		}

		entityIdToComponentIds.erase(it1);
	}
	else{
		std::string message = "Error: Could not find entity with id " + entityId.toString() + " when trying to delete\n";
		Log::error(message.c_str());
		return;
	}

	// remove from id to global index map
	std::map<Guid, int>::iterator it2 = idToGlobalIndex.find(entityId);
	if(it2 != idToGlobalIndex.end()){
		int index = it2->second;

		Entity* swappedEntity = destroyInternalEntity(entityAllocator, index);

		idToGlobalIndex.erase(it2);

		if(swappedEntity != NULL){
			entityIdsMarkedMoved.push_back(std::make_pair(swappedEntity->entityId, idToGlobalIndex[swappedEntity->entityId]));

			idToGlobalIndex[swappedEntity->entityId] = index;
		}
	}
	else{
		std::string message = "Error: Could not find entity with id " + entityId.toString() + " when trying to delete from id to global index map\n";
		Log::error(message.c_str());
		return;
	}

	//remove from id to type map
	std::map<Guid, int>::iterator it3 = idToType.find(entityId);
	if (it3 != idToType.end()) {
		idToType.erase(it3);
	}
	else {
		std::string message = "Error: Could not find entity with id " + entityId.toString() + " when trying to delete from id to type map\n";
		Log::error(message.c_str());
		return;
	}
}

void World::latentDestroyComponent(Guid entityId, Guid componentId, int componentType)
{
	std::string message = "latent destroy component: " + entityId.toString() + " " + componentId.toString() + " " + std::to_string(componentType) + "\n";
	Log::error(message.c_str());
	componentIdsMarkedLatentDestroy.push_back(make_triple(entityId, componentId, componentType));
}

void World::immediateDestroyComponent(Guid entityId, Guid componentId, int componentType)
{
	std::map<Guid, std::vector<std::pair<Guid, int>>>::iterator it1 = entityIdToComponentIds.find(entityId);
	if(it1 != entityIdToComponentIds.end()){
		for(size_t i = 0; i < it1->second.size(); i++){
			if(it1->second[i].first == componentId){
				it1->second.erase(it1->second.begin() + i);
				break;
			}
		}
	}

	// remove from id to global index map
	std::map<Guid, int>::iterator it2 = idToGlobalIndex.find(componentId);
	if(it2 != idToGlobalIndex.end()){
		int index = it2->second;

		Component* swappedComponent = NULL;
		if(componentType < 20){
			swappedComponent = destroyInternalComponent(&componentAllocatorMap, componentType, index);
		}
		else{
			swappedComponent = destroyComponent(&componentAllocatorMap, componentType, index);
		}

		idToGlobalIndex.erase(it2);

		if(swappedComponent != NULL){
			Log::info("Non null swapp component found\n");
			componentIdsMarkedMoved.push_back(make_triple(swappedComponent->componentId, componentType, idToGlobalIndex[swappedComponent->componentId]));

			idToGlobalIndex[swappedComponent->componentId] = index;
		}
	}
	else{
		std::string message = "Error: component id " + componentId.toString() + " not found in map when trying to destroy\n";
		Log::error(message.c_str());
	} 

	//remove from id to type map
	std::map<Guid, int>::iterator it3 = idToType.find(componentId);
	if (it3 != idToType.end()) {
		idToType.erase(it3);
	}
	else {
		std::string message = "Error: Could not find component with id " + componentId.toString() + " when trying to delete from id to type map\n";
		Log::error(message.c_str());
		return;
	}
}

bool World::isMarkedForLatentDestroy(Guid id)
{
	for(size_t i = 0; i < entityIdsMarkedLatentDestroy.size(); i++){
		if(entityIdsMarkedLatentDestroy[i] == id){
			return true;
		}
	}

	for(size_t i = 0; i < componentIdsMarkedLatentDestroy.size(); i++){
		if(componentIdsMarkedLatentDestroy[i].second == id){
			return true;
		}
	}

	return false;
}

void World::clearIdsMarkedCreatedOrDestroyed()
{
	entityIdsMarkedCreated.clear();
	entityIdsMarkedLatentDestroy.clear();
	componentIdsMarkedCreated.clear();
	componentIdsMarkedLatentDestroy.clear();
}

void World::clearIdsMarkedMoved()
{
	entityIdsMarkedMoved.clear();
	componentIdsMarkedMoved.clear();
}

std::vector<std::pair<Guid, int>> World::getComponentsOnEntity(Guid entityId)
{
	std::vector<std::pair<Guid, int>> componentsOnEntity;

	std::map<Guid, std::vector<std::pair<Guid, int>>>::iterator it = entityIdToComponentIds.find(entityId);
	if(it != entityIdToComponentIds.end()){
		componentsOnEntity = it->second;
	}

	return componentsOnEntity;
}

std::vector<Guid> World::getEntityIdsMarkedCreated()
{
	return entityIdsMarkedCreated;
}

std::vector<Guid> World::getEntityIdsMarkedLatentDestroy()
{
	return entityIdsMarkedLatentDestroy;
}

std::vector<std::pair<Guid, int>> World::getEntityIdsMarkedMoved()
{
	return entityIdsMarkedMoved;
}
			
std::vector<triple<Guid, Guid, int>> World::getComponentIdsMarkedCreated()
{
	return componentIdsMarkedCreated;
}

std::vector<triple<Guid, Guid, int>> World::getComponentIdsMarkedLatentDestroy()
{
	return componentIdsMarkedLatentDestroy;
}

std::vector<triple<Guid, int, int>> World::getComponentIdsMarkedMoved()
{
	return componentIdsMarkedMoved;
}

Bounds* World::getWorldBounds()
{
	return &bounds;
}

// Octtree* World::getStaticPhysicsTree()
// {
// 	return &stree;
// }

// Octtree* World::getDynamicPhysicsTree()
// {
// 	return &dtree;
// }

UniformGrid* World::getStaticPhysicsGrid()
{
	return &sgrid;
}

bool World::raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance)
{
	Ray ray;

	ray.origin = origin;
	ray.direction = direction;

	return sgrid.intersect(ray) != NULL;// || dtree.intersect(ray) != NULL;
	// return stree.intersect(ray) != NULL || dtree.intersect(ray) != NULL;
}

// begin by only implementing for spheres first and later I will add for bounds, capsules etc
bool World::raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance, Collider** collider)
{
	Ray ray;

	ray.origin = origin;
	ray.direction = direction;

	// Object* object = stree.intersect(ray);
	BoundingSphere* boundingSphere = sgrid.intersect(ray);

	if(boundingSphere != NULL){
		//std::cout << "AAAAAA id: " << boundingSphere->id.toString() << std::endl;
		std::map<Guid, int>::iterator it = idToGlobalIndex.find(boundingSphere->id);
		if(it != idToGlobalIndex.end()){
			int colliderIndex = it->second;

			if(boundingSphere->primitiveType == 0){
				*collider = getComponentByIndex<SphereCollider>(colliderIndex);
			}
			else if(boundingSphere->primitiveType == 1){
				*collider = getComponentByIndex<BoxCollider>(colliderIndex);
			}
			else{
				*collider = getComponentByIndex<MeshCollider>(colliderIndex);
			}
			return true;
		}
		else{
			std::cout << "Error: component id does not correspond to a global index" << std::endl;
			return false;
		}
	}

	return false;
}