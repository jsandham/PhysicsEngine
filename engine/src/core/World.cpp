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
		size_t size;
		assetBundleFile.read(reinterpret_cast<char*>(&size), sizeof(size_t));

		std::vector<char> data(size);
		assetBundleFile.read(reinterpret_cast<char*>(&data[0]), data.size() * sizeof(char));		

		int type = *reinterpret_cast<int*>(&data[0]);

		//std::cout << "type: " << type << " size: " << size <<  std::endl;

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
			asset = loadInternalAsset(data, &index);
		}
		else{
			asset = loadAsset(data, &index);
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
	    size_t size;
		sceneFile.read(reinterpret_cast<char*>(&size), sizeof(size_t));

		std::vector<char> data(size);
		sceneFile.read(reinterpret_cast<char*>(&data[0]), data.size() * sizeof(char));

		char classification = *reinterpret_cast<char*>(&data[0]);
		int type = *reinterpret_cast<int*>(&data[sizeof(char)]);
		//std::cout << "classification: " << classification << " type: " << type << " size: " << size << std::endl;

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
			if(type < 20){
				entity = loadInternalEntity(data, &index);
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
			int instanceType = -1;
			if(type < 20){
				component = loadInternalComponent(data, &index, &instanceType);
			}
			else{
				component = loadComponent(data, &index, &instanceType);
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
				system = loadInternalSystem(data, &index);
			}
			else{
				system = loadSystem(data, &index);
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


	//std::cout << "Number of systems: " << systems.size() << std::endl;

	// std::map<Guid, std::vector<std::pair<Guid, int>>>::iterator it;
	// for(it = entityIdToComponentIds.begin(); it != entityIdToComponentIds.end(); it++){
	// 	std::vector<std::pair<Guid, int>> temp = it->second;
	// 	std::cout << "Entity " << it->first.toString() << " has components: ";
	// 	for(int i = 0; i < temp.size(); i++){
	// 		std::cout << temp[i].first.toString() << " instance type: " << temp[i].second << " ";
	// 	}
	// 	std::cout << "" <<std::endl;
	// }

	// for(int i = 0; i < getNumberOfEntities(); i++){
	// 	Entity* entity = getEntityByIndex(i);

	// 	std::cout << "Entity id: " << entity->entityId.toString() << std::endl;

	// 	Transform* transform = getComponent<Transform>(entity->entityId);
	// 	if(transform != NULL){
	// 		std::cout << "Transform found with id: " << transform->componentId.toString() << std::endl;
	// 	}
	// }

	// Entity* entity = getAllocator<Entity>().get(0);

	// std::cout << "entity id: " << entity->entityId.toString() << std::endl;

	// std::cout << "count: " << getAllocator<Entity>().getCount() << std::endl;
	// std::cout << "count: " << getAllocator<Transform>().getCount() << std::endl;
	// std::cout << "count: " << getAllocator<Camera>().getCount() << std::endl;
	// std::cout << "count: " << getAllocator<Rigidbody>().getCount() << std::endl;
	// std::cout << "count: " << getAllocator<DirectionalLight>().getCount() << std::endl;


	// Rigidbody* r = addComponent<Rigidbody>(entity->entityId);

	// if(r != NULL){
	// 	std::cout << "Rigidbody " << r->componentId.toString() << " found on entity" << std::endl;
	// }

	// Rigidbody* rigidbody = getComponent<Rigidbody>(entity->entityId);

	// if(rigidbody != NULL){
	// 	std::cout << "Rigidbody " << rigidbody->componentId.toString() << " found on entity" << std::endl;

	// 	std::cout << rigidbody->useGravity << std::endl;
	// 	std::cout << rigidbody->mass << std::endl;
	// 	std::cout << rigidbody->drag << std::endl;
	// 	std::cout << rigidbody->angularDrag << std::endl;

	// 	std::cout << rigidbody->velocity.x << " " << rigidbody->velocity.y << " " << rigidbody->velocity.z << std::endl;
	// 	std::cout << rigidbody->centreOfMass.x << " " << rigidbody->centreOfMass.y << " " << rigidbody->centreOfMass.z << std::endl;
	// 	std::cout << rigidbody->angularVelocity.x << " " << rigidbody->angularVelocity.y << " " << rigidbody->angularVelocity.z << std::endl;

	// 	std::cout << rigidbody->halfVelocity.x << " " << rigidbody->halfVelocity.y << " " << rigidbody->halfVelocity.z << std::endl;
	// }

	// Texture2D* t1 = createAsset<Texture2D>();
	// Material* ma1 = createAsset<Material>();
	// Mesh* me1 = createAsset<Mesh>();
	// Texture2D* t2 = createAsset<Texture2D>();

	// std::cout << "Texture id: " << t1->assetId.toString() << std::endl;


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

void World::latentDestroy(Guid entityId)
{
	entityIdsMarkedForLatentDestroy.push_back(entityId);
}

void World::immediateDestroy(Guid entityId)
{
	Entity* entity = getEntity(entityId);

	if(entity == NULL){
	 	std::cout << "Error: Could not find entity (" << entityId.toString() << ") when calling immediateDestroy" << std::endl;
	 	return;
	}

	// std::vector<std::pair<Guid, int>> componentsOnEntity;

	// std::map<Guid, std::vector<std::pair, int>>>::iterator it = entityIdToComponentIds.find(entityId);
	// if(it != entityIdToComponentIds.end()){
	// 	componentsOnEntity = it->second;
	// }
	// else{
	// 	std::cout << "Error: Entity id (" << entityId.toString() << ") does not exist in entity id to component id map" << std::endl;
	// 	return;
	// }

	




	// int entityGlobalIndex = -1;
	// std::map<int, int>::iterator it1 = idToGlobalIndexMap.find(entityId);
	// if(it1 != idToGlobalIndexMap.end()){
	// 	entityGlobalIndex = it1->second;
	// }
	// else{
	// 	std::cout << "Error: When searching entity with id " << entityId << " no global index corresponding to this entity id was found" << std::endl;
	// 	return;
	// }

	// if(entityGlobalIndex < 0 || entityGlobalIndex >= numberOfEntities){
	// 	std::cout << "Error: Entity global index corresponding to entity with id " << entityId << " was out or range" << std::endl;
	// 	return;
	// }

	// for(int i = 0; i < 8; i++){
	// 	int componentId = entity->componentIds[i];
	// 	if(componentId != -1){
	// 		std::map<int, int>::iterator it2 = componentIdToTypeMap.find(componentId);
	// 		int componentType = -1;
	// 		if(it2 != componentIdToTypeMap.end()){
	// 			componentType = it2->second;
	// 		}
	// 		else{
	// 			std::cout << "Error: When searching entity with id " << entityId << " no component with id " << componentId << " was found in component type map" << std::endl;
	// 			return;
	// 		}

	// 		if(componentType == -1){
	// 			std::cout << "Error: When searching entity with id " << entityId << " the component type found corresponding to component " << componentId << " was invalid" << std::endl;
	// 			return;
	// 		}

	// 		int componentGlobalIndex = -1;
	// 		std::map<int, int>::iterator it3 = idToGlobalIndexMap.find(componentId);
	// 		if(it3 != idToGlobalIndexMap.end()){
	// 			componentGlobalIndex = it3->second;
	// 		}
	// 		else{
	// 			std::cout << "Error: When searching component with id " << componentId << " no global index corresponding to this component id was found" << std::endl;
	// 			return;
	// 		}

	// 		if(componentGlobalIndex < 0){
	// 			std::cout << "Error: Component global index corresponding to component with id " << componentId << " was out or range" << std::endl;
	// 			return;
	// 		}

	// 		if(componentType == Component::getInstanceType<Transform>()){
	// 			transforms[componentGlobalIndex] = transforms[numberOfTransforms - 1];
	// 			numberOfTransforms--;
	// 		}
	// 		else if(componentType == Component::getInstanceType<Rigidbody>()){
	// 			rigidbodies[componentGlobalIndex] = rigidbodies[numberOfRigidbodies- 1];
	// 			numberOfRigidbodies--;
	// 		}
	// 		else if(componentType == Component::getInstanceType<Camera>()){
	// 			cameras[componentGlobalIndex] = cameras[numberOfCameras - 1];
	// 			numberOfCameras--;
	// 		}
	// 		else if(componentType == Component::getInstanceType<MeshRenderer>()){
	// 			meshRenderers[componentGlobalIndex] = meshRenderers[numberOfMeshRenderers - 1];
	// 			numberOfMeshRenderers--;
	// 		}
	// 		else if(componentType == Component::getInstanceType<DirectionalLight>()){
	// 			directionalLights[componentGlobalIndex] = directionalLights[numberOfDirectionalLights - 1];
	// 			numberOfDirectionalLights--;
	// 		}
	// 		else if(componentType == Component::getInstanceType<SpotLight>()){
	// 			spotLights[componentGlobalIndex] = spotLights[numberOfSpotLights - 1];
	// 			numberOfSpotLights--;
	// 		}
	// 		else if(componentType == Component::getInstanceType<PointLight>()){
	// 			pointLights[componentGlobalIndex] = pointLights[numberOfPointLights - 1];
	// 			numberOfPointLights--;
	// 		}
	// 		else if(componentType == Component::getInstanceType<SphereCollider>()){
	// 			sphereColliders[componentGlobalIndex] = sphereColliders[numberOfSphereColliders - 1];
	// 			numberOfSphereColliders--;
	// 		}
	// 		else if(componentType == Component::getInstanceType<BoxCollider>()){
	// 			boxColliders[componentGlobalIndex] = boxColliders[numberOfBoxColliders - 1];
	// 			numberOfBoxColliders--;
	// 		}
	// 		else if(componentType == Component::getInstanceType<CapsuleCollider>()){
	// 			capsuleColliders[componentGlobalIndex] = capsuleColliders[numberOfCapsuleColliders - 1];
	// 			numberOfCapsuleColliders--;
	// 		}
	// 		else{
	// 			std::cout << "Error: Unknown component type found when calling immediateDestroy" << std::endl;
	// 			return;
	// 		}
	// 	}
	// }

	// entities[entityGlobalIndex] = entities[numberOfEntities - 1];
	// numberOfEntities--;
}

bool World::isMarkedForLatentDestroy(Guid entityId)
{
	for(unsigned int i = 0; i < entityIdsMarkedForLatentDestroy.size(); i++){
		if(entityIdsMarkedForLatentDestroy[i] == entityId){
			return true;
		}
	}

	return false;
}

std::vector<Guid> World::getEntitiesMarkedForLatentDestroy()
{
	return entityIdsMarkedForLatentDestroy;
}

Entity* World::instantiate()
{
	int globalIndex = (int)getAllocator<Entity>().getCount();
	Guid entityId = Guid::newGuid();

	idToGlobalIndex[entityId] = globalIndex;
	entityIdToComponentIds[entityId] = std::vector<std::pair<Guid, int>>();

	Entity* entity = create<Entity>();//new Entity;

	entity->entityId = entityId;

	return entity;
}

Entity* World::instantiate(Guid entityId)
{
	std::cout << "BBBBBBBBBBBBBBB" << std::endl;

	Entity* oldEntity = getEntity(entityId);

	if(oldEntity == NULL){ 
		std::cout << "Error: Could not find entity (" << entityId.toString() << ") when trying to instantiate" << std::endl;
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

	Entity* newEntity = instantiate();

	// add components to new entity
	for(size_t i = 0; i < oldComponents.size(); i++){
		Guid oldComponentId = oldComponents[i].first;
		int oldComponentType = oldComponents[i].second;

		// TODO: How do I add the new component using the instance type and old component Id???
	}

	return NULL;
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
		std::cout << "AAAAAA id: " << boundingSphere->id.toString() << std::endl;
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


























bool World::writeToBMP(const std::string& filepath, std::vector<unsigned char>& data, int width, int height, int numChannels)
{
	if (numChannels != 1 && numChannels != 2 && numChannels != 3 && numChannels != 4){
		std::cout << "TextureLoader: Number of channels must be 1, 2, 3, or 4 where each channel is 8 bits" << std::endl;
		return false;
	}

	if (data.size() != width*height*numChannels){
		std::cout << data.size() << " " << width << " " << height << " " << numChannels << std::endl;
		std::cout << "TextureLoader: Data does not match width, height, and number of channels given" << std::endl;
		return false;
	}

	std::vector<unsigned char> formatedData;
	if (numChannels == 1){
		formatedData.resize(3 * width*height);
		for (int i = 0; i < width*height; i++){
			formatedData[3 * i] = data[i];
			formatedData[3 * i + 1] = data[i];
			formatedData[3 * i + 2] = data[i];
		}
		numChannels = 3;
	}
	else {
		formatedData.resize(numChannels * width * height);
		for (int i = 0; i < numChannels*width*height; i++){
			formatedData[i] = data[i];
		}
	}

	BMPHeader header = {};

	header.fileType = 0x4D42;
	header.fileSize = sizeof(BMPHeader) + (unsigned int)formatedData.size();
	header.bitmapOffset = sizeof(BMPHeader);
	header.size = sizeof(BMPHeader) - 14;
	header.width = width;
	header.height = height;
	header.planes = 1;
	header.bitsPerPixel = (unsigned short)(numChannels * 8);
	header.compression = 0;
	header.sizeOfBitmap = (unsigned int)formatedData.size();
	header.horizontalResolution = 0;
	header.verticalResolution = 0;
	header.colorsUsed = 0;
	header.colorsImportant = 0;

	FILE* file = fopen(filepath.c_str(), "wb");
	if (file){
		fwrite(&header, sizeof(BMPHeader), 1, file);
		fwrite(&formatedData[0], formatedData.size(), 1, file);
		fclose(file);
	}
	else{
		std::cout << "TextureLoader: Failed to open file " << filepath << " for writing" << std::endl;
		return false;
	}

	std::cout << "TextureLoader: Screen capture successful" << std::endl;

	return true;
}