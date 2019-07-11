#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_set>

#include "../../include/core/Load.h"
#include "../../include/core/LoadInternal.h"
#include "../../include/core/World.h"
#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

World::World()
{
	bounds.centre = glm::vec3(0.0f, 0.0f, 40.0f);
	bounds.size = 2.0f * glm::vec3(200.0f, 200.0f, 200.0f);

	//stree.create(bounds, 2, 5);
	//dtree.create(bounds, 2, 5);

	debug = false;
	debugView = 0;
}

World::~World()
{

}

bool World::load(Scene scene, AssetBundle assetBundle)
{
	std::cout << "Loading asset bundle: " << assetBundle.filepath << std::endl;

	std::ifstream assetBundleFile;
	assetBundleFile.open(assetBundle.filepath, std::ios::binary);

	if(!assetBundleFile.is_open()){
		std::cout << "Error: Failed to open asset bundle " << assetBundle.filepath << std::endl;
		return false;
	}
	
	AssetBundleHeader bundleHeader;
	assetBundleFile.read(reinterpret_cast<char*>(&bundleHeader), sizeof(AssetBundleHeader));

	while( assetBundleFile.peek() != EOF )
	{
		char classification;
		assetBundleFile.read(reinterpret_cast<char*>(&classification), sizeof(char));

		int type;
		assetBundleFile.read(reinterpret_cast<char*>(&type), sizeof(int));

		size_t size;
		assetBundleFile.read(reinterpret_cast<char*>(&size), sizeof(size_t));

		std::vector<char> data(size);
		assetBundleFile.read(reinterpret_cast<char*>(&data[0]), data.size() * sizeof(char));		

		if(type <= -1){
			std::cout << "Error: Type cannot be less than 0 when reading asset bundle file" << std::endl;
			return false;
		}

		if(size <= 0){
			std::cout << "Error: Size cannot be less than 1 when reading asset bundle file" << std::endl;
			return false;
		}

		int index = -1;
		Asset* asset = NULL;
		if(type < 20){
			asset = loadInternalAsset(data, type, &index);
		}
		else{
			asset = loadAsset(data, type, &index);
		}

		if(asset == NULL || index == -1){
			std::cout << "Error: Could not load asset" << std::endl;
			return false;
		}

		if(assetIdToGlobalIndex.find(asset->assetId) == assetIdToGlobalIndex.end()){
			assetIdToGlobalIndex[asset->assetId] = index;
		}
		else{
			std::cout << "Error: Asset with id " << asset->assetId.toString() << " already exists in map" << std::endl;
			return false;
		}
	}

	assetBundleFile.close();

	std::cout << "done loading assets" << std::endl;

	std::ifstream sceneFile;
	sceneFile.open(scene.filepath, std::ios::binary);

	if(!sceneFile.is_open()){
		std::cout << "Error: Failed to open scene file " << scene.filepath << std::endl;
		return false;
	}

	SceneHeader sceneHeader;
	sceneFile.read(reinterpret_cast<char*>(&sceneHeader), sizeof(SceneHeader));
	while( sceneFile.peek() != EOF )
	{
		char classification;
		sceneFile.read(reinterpret_cast<char*>(&classification), sizeof(char));

		int type;
		sceneFile.read(reinterpret_cast<char*>(&type), sizeof(int));

	    size_t size;
		sceneFile.read(reinterpret_cast<char*>(&size), sizeof(size_t));

		std::vector<char> data(size);

		sceneFile.read(reinterpret_cast<char*>(&data[0]), data.size() * sizeof(char));

		if(type <= -1){
			std::cout << "Error: Type cannot be less than 0 when reading scene file" << std::endl;
			return false;
		}

		if(size <= 0){
			std::cout << "Error: Size cannot be less than 1 when reading scene file" << std::endl;
			return false;
		}

		int index = -1;
		if(classification == 'e'){
			Entity* entity = NULL;
			if(type == 0){
				entity = loadInternalEntity(data, &index);
			}
			else{
				std::cout << "Error: Entity must be of type 0" << std::endl;
				return false;
			}

			if(entity == NULL || index == -1){
				std::cout << "Error: Could not load entity corresponding to type " << type << std::endl;
				return false;
			}

			if(idToGlobalIndex.find(entity->entityId) == idToGlobalIndex.end()){
				idToGlobalIndex[entity->entityId] = index;
			}
			else{
				std::cout << "Error: Entity with id " << entity->entityId.toString() << " already exists in map" << std::endl;
				return false;
			}
		}
		else if(classification == 'c'){
			Component* component = NULL;
			itype instanceType = -1;
			if(type < 20){
				component = loadInternalComponent(data, type, &index, &instanceType);
			}
			else{
				component = loadComponent(data, type, &index, &instanceType);
			}

			if(component == NULL || index == -1 || instanceType == -1){
				std::cout << "Error: Could not load component corresponding to type " << type << std::endl;
				return false;
			}

			if(idToGlobalIndex.find(component->componentId) == idToGlobalIndex.end()){
				idToGlobalIndex[component->componentId] = index;
			}
			else{
				std::cout << "Error: Component with id " << component->componentId.toString() << " already exists in map" << std::endl;
				return false;
			}

			//std::cout << "entity id: " << component->entityId.toString() << std::endl;
			entityIdToComponentIds[component->entityId].push_back(std::make_pair(component->componentId, instanceType));
		}
		else if(classification == 's'){
			System* system = NULL;
			if(type < 20){
				system = loadInternalSystem(data, type, &index);
			}
			else{
				system = loadSystem(data, type, &index);
			}

			if(system == NULL || index == -1){
				std::cout << "Error: Could not load system corresponding to type " << type << std::endl;
				return false;
			}

			// maybe set system in vector??
			systems.push_back(system);
		}
		else{
			std::cout << "Error: Classification must be \'e\' (entity), \'c\' (component), or \'s\' (system)" << std::endl;
			return false;
		}
	}

	// sort systems by order
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

	sceneFile.close();

	return true;
}

int World::getNumberOfEntities()
{
	return (int)getAllocator<Entity>().getCount();
}

int World::getNumberOfSystems()
{
	return (int)systems.size();
}

Entity* World::getEntity(Guid id)
{
	std::map<Guid, int>::iterator it = idToGlobalIndex.find(id);
	if(it != idToGlobalIndex.end()){
		return getAllocator<Entity>().get(it->second);
	}
	else{
		std::cout << "Error: No entity with id " << id.toString() << " was found" << std::endl;
		return NULL;
	}
}

System* World::getSystem(Guid id)
{
	return NULL;
}

Entity* World::getEntityByIndex(int index)
{
	return getAllocator<Entity>().get(index);
}

System* World::getSystemByIndex(int index)
{
	return systems[index];
}

int World::getIndexOf(Guid id)
{
	std::map<Guid, int>::iterator it2 = idToGlobalIndex.find(id);
	if( it2 != idToGlobalIndex.end()){
		return it2->second;
	}

	return -1;
}

int World::getIndexOfAsset(Guid id)
{
	std::map<Guid, int>::iterator it2 = assetIdToGlobalIndex.find(id);
	if( it2 != assetIdToGlobalIndex.end()){
		return it2->second;
	}

	return -1;
}

Entity* World::createEntity()
{
	int globalIndex = (int)getAllocator<Entity>().getCount();
	Guid entityId = Guid::newGuid();

	Entity* entity = create<Entity>();
	entity->entityId = entityId;

	idToGlobalIndex[entityId] = globalIndex;
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

	Entity* newEntity = createEntity();

	// add components to new entity
	for(size_t i = 0; i < oldComponents.size(); i++){
		Guid oldComponentId = oldComponents[i].first;
		int oldComponentType = oldComponents[i].second;

		// TODO: How do I add the new component using the instance type and old component Id???
	}

	return NULL;
}

void World::latentDestroyEntity(Guid entityId)
{
	entityIdsMarkedLatentDestroy.push_back(entityId);

	// add any components found on the entity to the latent destroy component list
	std::map<Guid, std::vector<std::pair<Guid, itype>>>::iterator it = entityIdToComponentIds.find(entityId);
	if(it != entityIdToComponentIds.end()){
		std::vector<std::pair<Guid, itype>> componentsOnEntity = it->second;
		for(size_t i = 0; i < componentsOnEntity.size(); i++){
			Guid componentId = componentsOnEntity[i].first;
			itype instanceType = componentsOnEntity[i].second;

			latentDestroyComponent(entityId, componentId, instanceType);
		}
	}
	else{
		std::cout << "Error: Could not find entity with id " << entityId.toString() << " when trying to add to latent destroy list" << std::endl;
		return;
	}
}

void World::immediateDestroyEntity(Guid entityId)
{
	std::map<Guid, std::vector<std::pair<Guid, itype>>>::iterator it1 = entityIdToComponentIds.find(entityId);
	if(it1 != entityIdToComponentIds.end()){
		std::vector<std::pair<Guid, itype>> componentsOnEntity = it1->second;

		for(size_t i = 0; i < componentsOnEntity.size(); i++){
			Guid componentId = componentsOnEntity[i].first;
			itype instanceType = componentsOnEntity[i].second;

			immediateDestroyComponent(entityId, componentId, instanceType);
		}

		entityIdToComponentIds.erase(it1);
	}
	else{
		std::cout << "Error: Could not find entity with id " << entityId.toString() << " when trying to delete" << std::endl;
		return;
	}

	std::map<Guid, int>::iterator it2 = idToGlobalIndex.find(entityId);
	if(it2 != idToGlobalIndex.end()){
		int index = it2->second;

		Entity* swappedEntity = destroyInternalEntity(index);

		idToGlobalIndex.erase(it2);

		if(swappedEntity != NULL){
			entityIdsMarkedMoved.push_back(std::make_pair(swappedEntity->entityId, idToGlobalIndex[swappedEntity->entityId]));

			idToGlobalIndex[swappedEntity->entityId] = index;
		}
	}
	else{
		std::cout << "Error: Could not find entity with id " << entityId.toString() << " when trying to delete" << std::endl;
		return;
	}
}

void World::latentDestroyComponent(Guid entityId, Guid componentId, itype instanceType)
{
	componentIdsMarkedLatentDestroy.push_back(make_triple(entityId, componentId, instanceType));
}

void World::immediateDestroyComponent(Guid entityId, Guid componentId, itype instanceType)
{
	std::map<Guid, std::vector<std::pair<Guid, itype>>>::iterator it1 = entityIdToComponentIds.find(entityId);
	if(it1 != entityIdToComponentIds.end()){
		for(size_t i = 0; i < it1->second.size(); i++){
			if(it1->second[i].first == componentId){
				it1->second.erase(it1->second.begin() + i);
			}
		}
	}

	std::map<Guid, int>::iterator it2 = idToGlobalIndex.find(componentId);
	if(it2 != idToGlobalIndex.end()){
		int index = it2->second;

		Component* swappedComponent = NULL;
		swappedComponent = destroyInternalComponent(instanceType, index);
		// if(instanceType < 20){
		// 	swappedComponent = destroyInternalComponent(instanceType, index);
		// }
		// else{
		// 	swappedComponent = destroyComponent(instanceType, index);
		// }

		idToGlobalIndex.erase(it2);

		if(swappedComponent != NULL){
			componentIdsMarkedMoved.push_back(make_triple(swappedComponent->componentId, instanceType, idToGlobalIndex[swappedComponent->componentId]));

			idToGlobalIndex[swappedComponent->componentId] = index;
		}
	}
	else{
		std::cout << "Error: component id " << componentId.toString() << " not found in map when trying to destroy" << std::endl;
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
			
std::vector<triple<Guid, Guid, itype>> World::getComponentIdsMarkedCreated()
{
	return componentIdsMarkedCreated;
}

std::vector<triple<Guid, Guid, itype>> World::getComponentIdsMarkedLatentDestroy()
{
	return componentIdsMarkedLatentDestroy;
}

std::vector<triple<Guid, itype, int>> World::getComponentIdsMarkedMoved()
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