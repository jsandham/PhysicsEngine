#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_set>

#include "../../include/core/Manager.h"
#include "../../include/core/Geometry.h"

#include "../../include/json/json.hpp" 
#include "../../include/stb_image/stb_image.h"

#include "../../include/systems/LoadSystem.h"
#include "../../include/systems/LoadInternalSystem.h"

using namespace json;
using namespace PhysicsEngine;

Manager::Manager()
{
	std::ifstream in("../data/build_settings.json", std::ios::in | std::ios::binary);
	std::ostringstream contents; contents << in.rdbuf(); in.close();

	json::JSON jsonBuildSettings = JSON::Load(contents.str());

	json::JSON::JSONWrapper<map<string,JSON>> objects = jsonBuildSettings.ObjectRange();
	map<string,JSON>::iterator it;

	for(it = objects.begin(); it != objects.end(); it++){
		if(it->first == "Settings"){
			settings.maxAllowedEntities = it->second["maxAllowedEntities"].ToInt();
			settings.maxAllowedTransforms = it->second["maxAllowedTransforms"].ToInt();
			settings.maxAllowedRigidbodies = it->second["maxAllowedRigidbodies"].ToInt();
			settings.maxAllowedCameras = it->second["maxAllowedCameras"].ToInt();
			settings.maxAllowedMeshRenderers = it->second["maxAllowedMeshRenderers"].ToInt();
			settings.maxAllowedLineRenderers = it->second["maxAllowedLineRenderers"].ToInt();
			settings.maxAllowedDirectionalLights = it->second["maxAllowedDirectionalLights"].ToInt();
			settings.maxAllowedSpotLights = it->second["maxAllowedSpotLights"].ToInt();
			settings.maxAllowedPointLights = it->second["maxAllowedPointLights"].ToInt();
			settings.maxAllowedBoxColliders = it->second["maxAllowedBoxColliders"].ToInt();
			settings.maxAllowedSphereColliders = it->second["maxAllowedSphereColliders"].ToInt();
			settings.maxAllowedCapsuleColliders = it->second["maxAllowedCapsuleColliders"].ToInt();

			settings.maxAllowedMaterials = it->second["maxAllowedMaterials"].ToInt();
			settings.maxAllowedTextures = it->second["maxAllowedTextures"].ToInt();
			settings.maxAllowedShaders = it->second["maxAllowedShaders"].ToInt();
			settings.maxAllowedMeshes = it->second["maxAllowedMeshes"].ToInt();
			settings.maxAllowedGMeshes = it->second["maxAllowedGMeshes"].ToInt();

			settings.physicsDepth = it->second["physicsDepth"].ToInt();

			settings.centre[0] = (float)it->second["centre"][0].ToFloat();
			settings.centre[1] = (float)it->second["centre"][1].ToFloat();
			settings.centre[2] = (float)it->second["centre"][2].ToFloat();

			settings.extent[0] = (float)it->second["extent"][0].ToFloat();
			settings.extent[1] = (float)it->second["extent"][1].ToFloat();
			settings.extent[2] = (float)it->second["extent"][2].ToFloat();
		}
	}

	bool error = settings.maxAllowedEntities <= 0;
	error |= settings.maxAllowedTransforms <= 0;
	error |= settings.maxAllowedRigidbodies <= 0;
	error |= settings.maxAllowedCameras <= 0;
	error |= settings.maxAllowedMeshRenderers <= 0;
	error |= settings.maxAllowedLineRenderers <= 0;
	error |= settings.maxAllowedDirectionalLights <= 0;
	error |= settings.maxAllowedSpotLights <= 0;
	error |= settings.maxAllowedPointLights <= 0;
	error |= settings.maxAllowedBoxColliders <= 0;
	error |= settings.maxAllowedSphereColliders <= 0;
	error |= settings.maxAllowedCapsuleColliders <= 0;

	error |= settings.maxAllowedMaterials <= 0;
	error |= settings.maxAllowedTextures <= 0;
	error |= settings.maxAllowedShaders <= 0;
	error |= settings.maxAllowedMeshes <= 0;
	error |= settings.maxAllowedGMeshes <= 0;

	if(error){
		std::cout << "Error: Max allowed values cannot be equal to or less than zero" << std::endl;
		return;
	}

	entities = new Pool<Entity>(settings.maxAllowedEntities);
	transforms = new Pool<Transform>(settings.maxAllowedTransforms);
	rigidbodies = new Pool<Rigidbody>(settings.maxAllowedRigidbodies);
	cameras = new Pool<Camera>(settings.maxAllowedCameras);
	meshRenderers = new Pool<MeshRenderer>(settings.maxAllowedMeshRenderers);
	lineRenderers = new Pool<LineRenderer>(settings.maxAllowedLineRenderers);
	directionalLights = new Pool<DirectionalLight>(settings.maxAllowedDirectionalLights);
	spotLights = new Pool<SpotLight>(settings.maxAllowedSpotLights);
	pointLights = new Pool<PointLight>(settings.maxAllowedPointLights);
	boxColliders = new Pool<BoxCollider>(settings.maxAllowedBoxColliders);
	sphereColliders = new Pool<SphereCollider>(settings.maxAllowedSphereColliders);
	capsuleColliders = new Pool<CapsuleCollider>(settings.maxAllowedCapsuleColliders);

	materials = new Pool<Material>(settings.maxAllowedMaterials);
	textures = new Pool<Texture2D>(settings.maxAllowedTextures);
	shaders = new Pool<Shader>(settings.maxAllowedShaders);
	meshes = new Pool<Mesh>(settings.maxAllowedMeshes);
	gmeshes = new Pool<GMesh>(settings.maxAllowedGMeshes);

	line = new Line();

	glm::vec3 centre = glm::vec3(settings.centre[0], settings.centre[1], settings.centre[2]);
	glm::vec3 size = 2.0f * glm::vec3(settings.extent[0], settings.extent[1], settings.extent[2]);

	bounds = new Bounds(centre, size);
	physics = new Octtree(*bounds, /*settings.physicsDepth*/2);

	componentTypeToPool[Component::getInstanceType<Transform>()] = reinterpret_cast<Pool<Transform>*>(transforms);
	componentTypeToPool[Component::getInstanceType<Rigidbody>()] = reinterpret_cast<Pool<Rigidbody>*>(rigidbodies);
	componentTypeToPool[Component::getInstanceType<Camera>()] = reinterpret_cast<Pool<Camera>*>(cameras);
	componentTypeToPool[Component::getInstanceType<MeshRenderer>()] = reinterpret_cast<Pool<MeshRenderer>*>(meshRenderers);
	componentTypeToPool[Component::getInstanceType<LineRenderer>()] = reinterpret_cast<Pool<LineRenderer>*>(lineRenderers);
	componentTypeToPool[Component::getInstanceType<DirectionalLight>()] = reinterpret_cast<Pool<DirectionalLight>*>(directionalLights);
	componentTypeToPool[Component::getInstanceType<SpotLight>()] = reinterpret_cast<Pool<SpotLight>*>(spotLights);
	componentTypeToPool[Component::getInstanceType<PointLight>()] = reinterpret_cast<Pool<PointLight>*>(pointLights);
	componentTypeToPool[Component::getInstanceType<BoxCollider>()] = reinterpret_cast<Pool<BoxCollider>*>(boxColliders);
	componentTypeToPool[Component::getInstanceType<SphereCollider>()] = reinterpret_cast<Pool<SphereCollider>*>(sphereColliders);
	componentTypeToPool[Component::getInstanceType<CapsuleCollider>()] = reinterpret_cast<Pool<CapsuleCollider>*>(capsuleColliders);

	assetTypeToPool[Asset::getInstanceType<Material>()] = reinterpret_cast<Pool<Material>*>(materials);
	assetTypeToPool[Asset::getInstanceType<Texture2D>()] = reinterpret_cast<Pool<Texture2D>*>(textures);
	assetTypeToPool[Asset::getInstanceType<Shader>()] = reinterpret_cast<Pool<Shader>*>(shaders);
	assetTypeToPool[Asset::getInstanceType<Mesh>()] = reinterpret_cast<Pool<Mesh>*>(meshes);
	assetTypeToPool[Asset::getInstanceType<GMesh>()] = reinterpret_cast<Pool<GMesh>*>(gmeshes);

	debug = false;
}

Manager::~Manager()
{
	delete entities;
	delete transforms;
	delete rigidbodies;
	delete cameras;
	delete meshRenderers;
	delete lineRenderers;
	delete directionalLights;
	delete spotLights;
	delete pointLights;
	delete boxColliders;
	delete sphereColliders;
	delete capsuleColliders;

	delete materials;
	delete shaders;
	delete textures;
	delete meshes;
	delete gmeshes;

	delete line;

	delete bounds;
	delete physics;

	for(unsigned int i = 0; i < systems.size(); i++){
		delete systems[i];
	}
}

// bool Manager::validate(std::vector<Scene> scenes, std::vector<AssetFile> assetFiles)
// {
// 	std::vector<Guid> materialShaderIds;
// 	std::vector<Guid> materialTextureIds;

// 	// check that all asset id's are unique and that the shader and texture ids on materials match actual shaders and textures
// 	for(unsigned int i = 0; i < assetFiles.size(); i++){
// 		std::string jsonAssetFilePath = assetFiles[i].filepath.substr(0, assetFiles[i].filepath.find_last_of(".")) + ".json";
// 		std::ifstream in(jsonAssetFilePath, std::ios::in | std::ios::binary);
// 		std::ostringstream contents; contents << in.rdbuf(); in.close();

// 		json::JSON jsonAsset = JSON::Load(contents.str());

// 		Guid assetId = jsonAsset["id"].ToString();
		
// 		if(assetIdToFilePath.count(assetId) == 0){
// 			assetIdToFilePath[assetId] = assetFiles[i].filepath;
// 		}
// 		else{
// 			std::cout << "Error: Duplicate asset id (" << assetId.toString() << ") exists" << std::endl;
// 			return false;
// 		}

// 		if(assetFiles[i].filepath.substr(assetFiles[i].filepath.find_last_of(".") + 1) == "mat"){
// 			Guid shaderId = jsonAsset["shader"].ToString();
// 			Guid mainTextureId = jsonAsset["mainTexture"].ToString();
// 			Guid normalMapId = jsonAsset["normalMap"].ToString();
// 			Guid specularMapId = jsonAsset["specularMap"].ToString();

// 			std::cout << "shaderId: " << shaderId.toString() << " mainTextureId: " << mainTextureId.toString() << " normalMapId: " << normalMapId.toString() << " specularMapId: " << specularMapId.toString() << std::endl;

// 			if(shaderId != Guid::INVALID) { materialShaderIds.push_back(shaderId); }
// 			if(mainTextureId != Guid::INVALID) { materialTextureIds.push_back(mainTextureId); }
// 			if(normalMapId != Guid::INVALID) { materialTextureIds.push_back(normalMapId); }
// 			if(specularMapId != Guid::INVALID) { materialTextureIds.push_back(specularMapId); }
// 		}
// 	}

// 	for(unsigned int i = 0; i < materialShaderIds.size(); i++){
// 		std::map<Guid, std::string>::iterator it = assetIdToFilePath.find(materialShaderIds[i]);
// 		std::cout << "shader id: " << materialShaderIds[i].toString() << std::endl;
// 		if(it != assetIdToFilePath.end()){
// 			std::string filepath = it->second;
// 			std::cout << "shader path: " << filepath << std::endl;
// 			if(filepath.substr(filepath.find_last_of(".") + 1) != "shader"){
// 				std::cout << "Error: Shader id found inside material does not correspond to a shader" << std::endl;
// 				return false;
// 			}
// 		}
// 		else{
// 			std::cout << "Error: Shader id in material does not match any asset file path" << std::endl;
// 			return false;
// 		}
// 	}

// 	for(unsigned int i = 0; i < materialTextureIds.size(); i++){
// 		std::cout << "texture id: " << materialTextureIds[i].toString() << std::endl;
// 		std::map<Guid, std::string>::iterator it = assetIdToFilePath.find(materialTextureIds[i]);
// 		if(it != assetIdToFilePath.end()){
// 			std::string filepath = it->second;
// 			std::cout << "texture: " << filepath << std::endl;
// 			if(filepath.substr(filepath.find_last_of(".") + 1) != "png"){
// 				std::cout << "Error: Texture id found inside material does not correspond to a texture" << std::endl;
// 				return false;
// 			}
// 		}
// 		else{
// 			std::cout << "Error: Texture id in material does not match any asset file path" << std::endl;
// 			return false;
// 		}
// 	}

// 	// check that all entities and components have unique ids accross all scenes and that all scene ids are unique
// 	// std::unordered_set<Guid> sceneIds;
// 	// std::unordered_set<Guid> entityIds;
// 	std::map<Guid, Guid> sceneIds;
// 	std::map<Guid, Guid> entityIds;
// 	std::map<Guid, Guid> componentIdToEntityId;
// 	for(unsigned int i = 0; i < scenes.size(); i++){
// 		std::cout << "validating scene: " << scenes[i].name << std::endl;

// 		std::string jsonSceneFilePath = scenes[i].filepath.substr(0, scenes[i].filepath.find_last_of(".")) + ".json";
// 		std::ifstream in(jsonSceneFilePath, std::ios::in | std::ios::binary);
// 		std::ostringstream contents; contents << in.rdbuf(); in.close();

// 		json::JSON jsonScene = JSON::Load(contents.str());

// 		json::JSON::JSONWrapper<map<string,JSON>> objects = jsonScene.ObjectRange();
// 		map<string,JSON>::iterator it;

// 		for(it = objects.begin(); it != objects.end(); it++){
// 			if(it->first == "id"){ 
// 				Guid sceneId = it->second.ToString();//////////////////////////////////////
// 				if(sceneIds.find(sceneId) == sceneIds.end()){
// 					sceneIds.insert(std::pair<Guid, Guid>(sceneId, sceneId));
// 					continue;
// 				} 
// 				else{
// 					std::cout << "Error: Duplicate scene id found" << std::endl;
// 					return false;
// 				}
// 			}

// 			Guid objectId = it->first; //////////////////////////
// 			std::string type = it->second["type"].ToString();

// 			if(type == "Entity"){
// 				if (entityIds.find(objectId) == entityIds.end()) {
// 					entityIds.insert(std::pair<Guid, Guid>(objectId, objectId));
// 				}
// 				else{
// 					std::cout << "Error: Duplicate entity id found" << std::endl;
// 					return false;
// 				}
// 			}
// 			// else if(type == "MeshRenderer"){
// 			// 	int entityId = it->second["entity"].ToInt();
// 			// 	if(componentIdToEntityIdMap.count(objectId) == 0){
// 			// 		componentIdToEntityIdMap[objectId] = entityId;
// 			// 	}
// 			// 	else{
// 			// 		std::cout << "Error: Duplicate component ids exist" << std::endl;
// 			// 		return false;
// 			// 	}
				
// 			// 	int meshId = it->second["mesh"].ToInt();
// 			// 	int materialId = it->second["material"].ToInt();

// 			// 	if(assetIdToFilePathMap.count(meshId) != 1){
// 			// 		std::cout << "Error: Mesh id (" << meshId << ") found on MeshRenderer does not match a mesh" << std::endl;
// 			// 		return false;
// 			// 	}

// 			// 	if(assetIdToFilePathMap.count(materialId) != 1){
// 			// 		std::cout << "Error: Material id (" << materialId << ") found on MeshRenderer does not match a material" << std::endl;
// 			// 		return false;
// 			// 	}
// 			// }


// 			else if(type == "PhysicsSystem"){
// 				std::cout << "physics system found" << std::endl;
// 			}
// 			else if(type == "RenderSystem"){
// 				std::cout << "render system found" << std::endl;
// 			}
// 			else if(type == "LogicSystem"){
// 				std::cout << "logic system found" << std::endl;
// 			}
// 			else if(type == "PlayerSystem"){
// 				std::cout << "player system found" << std::endl;
// 			}
// 			else if(type == "DebugSystem"){
// 				std::cout << "debug system found" << std::endl;
// 			}
// 			else{
// 				Guid entityId = it->second["entity"].ToString();
// 				if(componentIdToEntityId.count(objectId) == 0){
// 					componentIdToEntityId[objectId] = entityId;
// 				}
// 				else{
// 					std::cout << "Error: Duplicate component ids exist" << std::endl;
// 					return false;
// 				}

// 				entityIdToComponentIds[entityId].push_back(objectId);

// 				if(type == "MeshRenderer"){
// 					Guid meshId = it->second["mesh"].ToString();
// 					Guid materialId = it->second["material"].ToString();

// 					if(assetIdToFilePath.count(meshId) != 1){
// 						std::cout << "Error: Mesh id (" << meshId.toString() << ") found on MeshRenderer does not match a mesh" << std::endl;
// 						return false;
// 					}

// 					if(assetIdToFilePath.count(materialId) != 1){
// 						std::cout << "Error: Material id (" << materialId.toString() << ") found on MeshRenderer does not match a material" << std::endl;
// 						return false;
// 					}
// 				}

// 				if(type == "LineRenderer"){
// 					Guid materialId = it->second["material"].ToString();

// 					if(assetIdToFilePath.count(materialId) != 1){
// 						std::cout << "Error: Material id (" << materialId.toString() << ") found on LineRenderer does not match a material" << std::endl;
// 						return false;
// 					}
// 				}
// 			}
// 		}

// 		// check that every components entity id matches an existing entity
// 		std::map<Guid, Guid>::iterator iter;
// 		for(iter = componentIdToEntityId.begin(); iter != componentIdToEntityId.end(); iter++){
// 			Guid entityId = iter->second;
// 			if(entityIds.find(entityId) == entityIds.end()){
// 				std::cout << "Error: Component says it is attached to an entity that does not exist" << std::endl;
// 				return false;
// 			}
// 		}
// 	}

// 	return true;
// }

bool Manager::load(Scene scene, std::vector<AssetFile> assetFiles)
{
	std::cout << "loading scene: " << scene.name << std::endl;

	for(unsigned int i = 0; i < assetFiles.size(); i++){
		std::string jsonAssetFilePath = assetFiles[i].filepath.substr(0, assetFiles[i].filepath.find_last_of(".")) + ".json";
		std::ifstream in(jsonAssetFilePath, std::ios::in | std::ios::binary);
		std::ostringstream contents; contents << in.rdbuf(); in.close();

		json::JSON jsonAsset = JSON::Load(contents.str());

		Guid assetId = jsonAsset["id"].ToString();

		std::map<Guid, std::string>::iterator it = assetIdToFilePath.find(assetId);
		if(it == assetIdToFilePath.end()){
			assetIdToFilePath[assetId] = assetFiles[i].filepath;
			std::cout << "asset file: " << assetFiles[i].filepath << std::endl;
		}
	}

	std::cout << "scene file path: " << scene.filepath << std::endl;

	std::string binarySceneFilePath = scene.filepath.substr(0, scene.filepath.find_last_of(".")) + ".scene";

	SceneHeader sceneHeader = {};
	FILE* file = fopen(binarySceneFilePath.c_str(), "rb");
	size_t bytesRead;
	if (file){
		bytesRead = fread(&sceneHeader, sizeof(SceneHeader), 1, file);

		for(unsigned int i = 0; i < systems.size(); i++){ delete systems[i]; }

		systems.clear(); 

		std::cout << "de-serialized scene header file contains the following information: " << std::endl;

		std::cout << "sizeOfEntity: " << sceneHeader.sizeOfEntity << std::endl;
		std::cout << "sizeOfTransform: " << sceneHeader.sizeOfTransform << std::endl;
		std::cout << "sizeOfRigidbodies: " << sceneHeader.sizeOfRigidbody << std::endl;
		std::cout << "sizeOfCameras: " << sceneHeader.sizeOfCamera << std::endl;
		std::cout << "sizeOfMeshRenderer: " << sceneHeader.sizeOfMeshRenderer << std::endl;
		std::cout << "sizeOfLineRenderer: " << sceneHeader.sizeOfLineRenderer << std::endl;
		std::cout << "sizeOfDirectionalLight: " << sceneHeader.sizeOfDirectionalLight << std::endl;
		std::cout << "sizeOfSpotLight: " << sceneHeader.sizeOfSpotLight << std::endl;
		std::cout << "sizeOfPointLight: " << sceneHeader.sizeOfPointLight << std::endl;
		std::cout << "sizeOfBoxCollider: " << sceneHeader.sizeOfBoxCollider << std::endl;
		std::cout << "sizeOfSphereCollider: " << sceneHeader.sizeOfSphereCollider << std::endl;
		std::cout << "sizeOfCapsuleCollider: " << sceneHeader.sizeOfCapsuleCollider << std::endl;

		int existingNumberOfEntities = entities->getIndex();
		int existingNumberOfTransforms = transforms->getIndex();
		int existingNumberOfRigidbodies = rigidbodies->getIndex();
		int existingNumberOfCameras = cameras->getIndex();
		int existingNumberOfMeshRenderers = meshRenderers->getIndex();
		int existingNumberOfLineRenderers = lineRenderers->getIndex();
		int existingNumberOfDirectionalLights = directionalLights->getIndex();
		int existingNumberOfSpotLights = spotLights->getIndex();
		int existingNumberOfPointLights = pointLights->getIndex();
		int existingNumberOfBoxColliders = boxColliders->getIndex();
		int existingNumberOfSphereColliders = sphereColliders->getIndex();
		int existingNumberOfCapsuleColliders = capsuleColliders->getIndex();

		bool error = existingNumberOfEntities + sceneHeader.numberOfEntities > settings.maxAllowedEntities;
		error |= existingNumberOfTransforms + sceneHeader.numberOfTransforms > settings.maxAllowedTransforms;
		error |= existingNumberOfRigidbodies + sceneHeader.numberOfRigidbodies > settings.maxAllowedRigidbodies;
		error |= existingNumberOfCameras + sceneHeader.numberOfCameras > settings.maxAllowedCameras;
		error |= existingNumberOfMeshRenderers + sceneHeader.numberOfMeshRenderers > settings.maxAllowedMeshRenderers;
		error |= existingNumberOfLineRenderers + sceneHeader.numberOfLineRenderers > settings.maxAllowedLineRenderers;
		error |= existingNumberOfDirectionalLights + sceneHeader.numberOfDirectionalLights > settings.maxAllowedDirectionalLights;
		error |= existingNumberOfSpotLights + sceneHeader.numberOfSpotLights > settings.maxAllowedSpotLights;
		error |= existingNumberOfPointLights + sceneHeader.numberOfPointLights > settings.maxAllowedPointLights;
		error |= existingNumberOfBoxColliders + sceneHeader.numberOfBoxColliders > settings.maxAllowedBoxColliders;
		error |= existingNumberOfSphereColliders + sceneHeader.numberOfSphereColliders > settings.maxAllowedSphereColliders;
		error |= existingNumberOfCapsuleColliders + sceneHeader.numberOfCapsuleColliders > settings.maxAllowedCapsuleColliders;

		if(error){
			std::cout << "Error: Number of entities, components, or systems exceeds maximum allowed. Please increase max allowed in scene settings." << std::endl;
			return false;
		}

		error = settings.maxAllowedEntities <= 0;
		error |= settings.maxAllowedTransforms <= 0;
		error |= settings.maxAllowedRigidbodies <= 0;
		error |= settings.maxAllowedCameras <= 0;
		error |= settings.maxAllowedMeshRenderers <= 0;
		error |= settings.maxAllowedLineRenderers <= 0;
		error |= settings.maxAllowedDirectionalLights <= 0;
		error |= settings.maxAllowedSpotLights <= 0;
		error |= settings.maxAllowedPointLights <= 0;
		error |= settings.maxAllowedBoxColliders <= 0;
		error |= settings.maxAllowedSphereColliders <= 0;
		error |= settings.maxAllowedCapsuleColliders <= 0;

		if(error){
			std::cout << "Error: Total number of entities, components, and systems must be strictly greater than zero. Please increase max allowed in scene settings." << std::endl;
			return false;
		}

		if(sceneHeader.numberOfEntities > 0){
			EntityData* entityData = new EntityData[sceneHeader.numberOfEntities];
			bytesRead = fread(entityData, sceneHeader.numberOfEntities*sizeof(EntityData), 1, file);
			for(unsigned int i = 0; i < sceneHeader.numberOfEntities; i++){ 
				entities->get(i)->load(entityData[i]);
				entities->get(i)->setManager(this); 

				idToGlobalIndex[entities->get(i)->entityId] = i;
			}

			entities->setIndex(sceneHeader.numberOfEntities);

			delete [] entityData; 
		}

		if(sceneHeader.numberOfTransforms > 0){
			TransformData* transformData = new TransformData[sceneHeader.numberOfTransforms];
			bytesRead = fread(transformData, sceneHeader.numberOfTransforms*sizeof(TransformData), 1, file);
			for(unsigned int i = 0; i < sceneHeader.numberOfTransforms; i++){ 
				transforms->get(i)->load(transformData[i]);
				transforms->get(i)->setManager(this); 

				idToGlobalIndex[transforms->get(i)->componentId] = i;
				componentIdToType[transforms->get(i)->componentId] = Component::getInstanceType<Transform>(); 	
				entityIdToComponentIds[transforms->get(i)->entityId].push_back(transforms->get(i)->componentId);			
			}

			transforms->setIndex(sceneHeader.numberOfTransforms);

			delete [] transformData; 
		}

		if(sceneHeader.numberOfRigidbodies > 0){
			RigidbodyData* rigidbodyData = new RigidbodyData[sceneHeader.numberOfRigidbodies];
			bytesRead = fread(rigidbodyData, sceneHeader.numberOfRigidbodies*sizeof(RigidbodyData), 1, file);
			for(unsigned int i = 0; i < sceneHeader.numberOfRigidbodies; i++){
				rigidbodies->get(i)->load(rigidbodyData[i]); 
				rigidbodies->get(i)->setManager(this); 

				idToGlobalIndex[rigidbodies->get(i)->componentId] = i;
				componentIdToType[rigidbodies->get(i)->componentId] = Component::getInstanceType<Rigidbody>(); 
				entityIdToComponentIds[rigidbodies->get(i)->entityId].push_back(rigidbodies->get(i)->componentId);
			}

			rigidbodies->setIndex(sceneHeader.numberOfRigidbodies);

			delete [] rigidbodyData; 
		}	

		if(sceneHeader.numberOfCameras > 0){
			CameraData* cameraData = new CameraData[sceneHeader.numberOfCameras];
			bytesRead = fread(cameraData, sceneHeader.numberOfCameras*sizeof(CameraData), 1, file);
			for(unsigned int i = 0; i < sceneHeader.numberOfCameras; i++){
				cameras->get(i)->load(cameraData[i]); 
				cameras->get(i)->setManager(this); 

				idToGlobalIndex[cameras->get(i)->componentId] = i;
				componentIdToType[cameras->get(i)->componentId] = Component::getInstanceType<Camera>(); 
				entityIdToComponentIds[cameras->get(i)->entityId].push_back(cameras->get(i)->componentId);
			}

			cameras->setIndex(sceneHeader.numberOfCameras);

			delete [] cameraData; 
		}	

		if(sceneHeader.numberOfMeshRenderers > 0){
			MeshRendererData* meshRendererData = new MeshRendererData[sceneHeader.numberOfMeshRenderers];
			bytesRead = fread(meshRendererData, sceneHeader.numberOfMeshRenderers*sizeof(MeshRendererData), 1, file);
			for(unsigned int i = 0; i < sceneHeader.numberOfMeshRenderers; i++){ 
				meshRenderers->get(i)->load(meshRendererData[i]);
				meshRenderers->get(i)->setManager(this); 

				idToGlobalIndex[meshRenderers->get(i)->componentId] = i;
				componentIdToType[meshRenderers->get(i)->componentId] = Component::getInstanceType<MeshRenderer>(); 
				entityIdToComponentIds[meshRenderers->get(i)->entityId].push_back(meshRenderers->get(i)->componentId);
			}

			meshRenderers->setIndex(sceneHeader.numberOfMeshRenderers);

			delete [] meshRendererData; 
		}	

		if(sceneHeader.numberOfLineRenderers > 0){
			LineRendererData* lineRendererData = new LineRendererData[sceneHeader.numberOfLineRenderers];
			bytesRead = fread(lineRendererData, sceneHeader.numberOfLineRenderers*sizeof(LineRendererData), 1, file);
			for(unsigned int i = 0; i < sceneHeader.numberOfLineRenderers; i++){ 
				lineRenderers->get(i)->load(lineRendererData[i]);
				lineRenderers->get(i)->setManager(this); 

				idToGlobalIndex[lineRenderers->get(i)->componentId] = i;
				componentIdToType[lineRenderers->get(i)->componentId] = Component::getInstanceType<LineRenderer>(); 
				entityIdToComponentIds[lineRenderers->get(i)->entityId].push_back(lineRenderers->get(i)->componentId);
			}

			lineRenderers->setIndex(sceneHeader.numberOfLineRenderers);

			delete [] lineRendererData; 
		}

		if(sceneHeader.numberOfDirectionalLights > 0){
			DirectionalLightData* directionalLightData = new DirectionalLightData[sceneHeader.numberOfDirectionalLights];
			bytesRead = fread(directionalLightData, sceneHeader.numberOfDirectionalLights*sizeof(DirectionalLightData), 1, file);
			for(unsigned int i = 0; i < sceneHeader.numberOfDirectionalLights; i++){
				directionalLights->get(i)->load(directionalLightData[i]);
				directionalLights->get(i)->setManager(this);

				idToGlobalIndex[directionalLights->get(i)->componentId] = i;
				componentIdToType[directionalLights->get(i)->componentId] = Component::getInstanceType<DirectionalLight>(); 
				entityIdToComponentIds[directionalLights->get(i)->entityId].push_back(directionalLights->get(i)->componentId);
			}

			directionalLights->setIndex(sceneHeader.numberOfDirectionalLights);

			delete [] directionalLightData; 
		}	

		if(sceneHeader.numberOfSpotLights > 0){
			SpotLightData* spotLightData = new SpotLightData[sceneHeader.numberOfSpotLights];
			bytesRead = fread(spotLightData, sceneHeader.numberOfSpotLights*sizeof(SpotLightData), 1, file);
			for(unsigned int i = 0; i < sceneHeader.numberOfSpotLights; i++){
				spotLights->get(i)->load(spotLightData[i]);
				spotLights->get(i)->setManager(this); 

				idToGlobalIndex[spotLights->get(i)->componentId] = i;
				componentIdToType[spotLights->get(i)->componentId] = Component::getInstanceType<SpotLight>(); 
				entityIdToComponentIds[spotLights->get(i)->entityId].push_back(spotLights->get(i)->componentId);
			}

			spotLights->setIndex(sceneHeader.numberOfSpotLights);

			delete [] spotLightData; 
		}	

		if(sceneHeader.numberOfPointLights > 0){
			PointLightData* pointLightData = new PointLightData[sceneHeader.numberOfPointLights];
			bytesRead = fread(pointLightData, sceneHeader.numberOfPointLights*sizeof(PointLightData), 1, file);
			for(unsigned int i = 0; i < sceneHeader.numberOfPointLights; i++){
				pointLights->get(i)->load(pointLightData[i]);
				pointLights->get(i)->setManager(this); 

				idToGlobalIndex[pointLights->get(i)->componentId] = i;
				componentIdToType[pointLights->get(i)->componentId] = Component::getInstanceType<PointLight>(); 
				entityIdToComponentIds[pointLights->get(i)->entityId].push_back(pointLights->get(i)->componentId);
			}

			pointLights->setIndex(sceneHeader.numberOfPointLights);

			delete [] pointLightData; 
		}		

		if(sceneHeader.numberOfBoxColliders > 0){
			BoxColliderData* boxColliderData = new BoxColliderData[sceneHeader.numberOfBoxColliders];
			bytesRead = fread(boxColliderData, sceneHeader.numberOfBoxColliders*sizeof(BoxColliderData), 1, file);
			for(unsigned int i = 0; i < sceneHeader.numberOfBoxColliders; i++){
				boxColliders->get(i)->load(boxColliderData[i]);
				boxColliders->get(i)->setManager(this); 

				idToGlobalIndex[boxColliders->get(i)->componentId] = i;
				componentIdToType[boxColliders->get(i)->componentId] = Component::getInstanceType<BoxCollider>(); 
				entityIdToComponentIds[boxColliders->get(i)->entityId].push_back(boxColliders->get(i)->componentId);
			}

			boxColliders->setIndex(sceneHeader.numberOfBoxColliders);

			delete [] boxColliderData; 
		}	

		if(sceneHeader.numberOfSphereColliders > 0){
			SphereColliderData* sphereColliderData = new SphereColliderData[sceneHeader.numberOfSphereColliders];
			bytesRead = fread(sphereColliderData, sceneHeader.numberOfSphereColliders*sizeof(SphereColliderData), 1, file);
			for(unsigned int i = 0; i < sceneHeader.numberOfSphereColliders; i++){
				sphereColliders->get(i)->load(sphereColliderData[i]);
				sphereColliders->get(i)->setManager(this); 

				idToGlobalIndex[sphereColliders->get(i)->componentId] = i;
				componentIdToType[sphereColliders->get(i)->componentId] = Component::getInstanceType<SphereCollider>(); 
				entityIdToComponentIds[sphereColliders->get(i)->entityId].push_back(sphereColliders->get(i)->componentId);
			}

			sphereColliders->setIndex(sceneHeader.numberOfSphereColliders);

			delete [] sphereColliderData; 
		}	

		if(sceneHeader.numberOfCapsuleColliders > 0){
			CapsuleColliderData* capsuleColliderData = new CapsuleColliderData[sceneHeader.numberOfCapsuleColliders];
			bytesRead = fread(capsuleColliderData, sceneHeader.numberOfCapsuleColliders*sizeof(CapsuleColliderData), 1, file);
			for(unsigned int i = 0; i < sceneHeader.numberOfCapsuleColliders; i++){
				capsuleColliders->get(i)->load(capsuleColliderData[i]); 
				capsuleColliders->get(i)->setManager(this); 

				idToGlobalIndex[capsuleColliders->get(i)->componentId] = i;
				componentIdToType[capsuleColliders->get(i)->componentId] = Component::getInstanceType<CapsuleCollider>(); 
				entityIdToComponentIds[capsuleColliders->get(i)->entityId].push_back(capsuleColliders->get(i)->componentId);
			}

			capsuleColliders->setIndex(sceneHeader.numberOfCapsuleColliders);

			delete [] capsuleColliderData; 
		}		

		// what if I wrote the size of the system in the binary file, followed by the actual system byte data?
		while(true){
			size_t sizeOfSystem;
			bytesRead = fread(&sizeOfSystem, sizeof(size_t), 1, file);

			std::cout << "bytes read: " << bytesRead << " size of system data read: " << sizeOfSystem << std::endl;

			if(bytesRead != 1){
				break;
			}

			unsigned char* data = new unsigned char[sizeOfSystem];
			bytesRead = fread(data, sizeOfSystem * sizeof(unsigned char), 1, file);

			int type = *reinterpret_cast<int*>(data);

			std::cout << "type: " << type << std::endl;

			System* system = NULL;
			if(type < 10){
				system = loadInternalSystem(data);
			}
			else
			{
				system = loadSystem(data);
			}

			if(system == NULL){
				std::cout << "Error: Could not load system" << std::endl;
				return false;
			}

			system->setManager(this);

			systems.push_back(system);

			delete [] data;
		}

		fclose(file);
	}
	else{
		std::cout << "Error: Failed to open scene binary file " << binarySceneFilePath << " for reading" << std::endl;
		return false;
	}

	// find all unique materials
	std::vector<Guid> materialIds;
	for(int i = 0; i < meshRenderers->getIndex(); i++){
		bool materialIdFound = false;
		for(unsigned int j = 0; j < materialIds.size(); j++){
			if(meshRenderers->get(i)->materialId == materialIds[j]){
				materialIdFound = true;
				break;
			}
		}

		if(!materialIdFound){
			std::cout << "material id: " << meshRenderers->get(i)->materialId.toString() << " i: " << i << std::endl;
			materialIds.push_back(meshRenderers->get(i)->materialId);
		}
	}

	for(int i = 0; i < lineRenderers->getIndex(); i++){
		bool materialIdFound = false;
		for(unsigned int j = 0; j < materialIds.size(); j++){
			if(lineRenderers->get(i)->materialId == materialIds[j]){
				materialIdFound = true;
				break;
			}
		}

		if(!materialIdFound){
			std::cout << "material id: " << lineRenderers->get(i)->materialId.toString() << " i: " << i << std::endl;
			materialIds.push_back(lineRenderers->get(i)->materialId);
		}
	}

	// de-serialize all unique materials found
	for(unsigned int i = 0; i < materialIds.size(); i++){
		Guid materialId = materialIds[i];

		assetIdToGlobalIndex[materialId] = i;

		std::string materialFilePath = assetIdToFilePath[materialId];

		std::cout << "material id: " << materialId.toString() << " material file path: " << materialFilePath << std::endl;

		FILE* file = fopen(materialFilePath.c_str(), "rb");
		size_t bytesRead;
		if (file){
			MaterialData data;
			bytesRead = fread(&data, sizeof(MaterialData), 1, file);
			materials->get(i)->load(data);
			// bytesRead = fread(materials->get(i), sizeof(Material), 1, file);
			std::cout << "number of bytes read from file: " << bytesRead << std::endl;
			materials->increment();
		}
		else{
			std::cout << "Error: Failed to open material binary file " << materialFilePath << " for reading" << std::endl;
			return false;
		}

		std::cout << "material id: " << materials->get(i)->assetId.toString() << " texture id: " << materials->get(i)->textureId.toString() << " shader id: " << materials->get(i)->shaderId.toString() << std::endl;
	}

	// set manager on materials
	for(int i = 0; i < materials->getIndex(); i++){
		materials->get(i)->setManager(this);
	}

	// find all unique textures and shaders 
	std::vector<Guid> textureIds;
	std::vector<Guid> shaderIds;
	for(unsigned int i = 0; i < materialIds.size(); i++){
		if(materials->get(i)->textureId != Guid::INVALID){
			bool mainTextureIdFound = false;
			for(unsigned int j = 0; j < textureIds.size(); j++){
				if(materials->get(i)->textureId == textureIds[j]){
					mainTextureIdFound = true;
					break;
				}
			}

			if(!mainTextureIdFound){
				textureIds.push_back(materials->get(i)->textureId);
			}
		}

		if(materials->get(i)->normalMapId != Guid::INVALID){
			bool normalMapIdFound = false;
			for(unsigned int j = 0; j < textureIds.size(); j++){
				if(materials->get(i)->normalMapId == textureIds[j]){
					normalMapIdFound = true;
					break;
				}
			}

			if(!normalMapIdFound){
				textureIds.push_back(materials->get(i)->normalMapId);
			}
		}

		if(materials->get(i)->specularMapId != Guid::INVALID){
			bool specularMapIdFound = false;
			for(unsigned int j = 0; j < textureIds.size(); j++){
				if(materials->get(i)->specularMapId == textureIds[j]){
					specularMapIdFound = true;
					break;
				}
			}

			if(!specularMapIdFound){
				textureIds.push_back(materials->get(i)->specularMapId);
			}
		}

		bool shaderIdFound = false;
		for(unsigned int j = 0; j < shaderIds.size(); j++){
			if(materials->get(i)->shaderId == shaderIds[j]){
				shaderIdFound = true;
				break;
			}
		}

		if(!shaderIdFound){
			shaderIds.push_back(materials->get(i)->shaderId);
		}
	}

	// de-serialize all unique textures and shaders found
	for(unsigned int i = 0; i < textureIds.size(); i++){
		Guid textureId = textureIds[i];

		std::cout << "textureId: " << textureId.toString() << " i: " << i << std::endl;

		assetIdToGlobalIndex[textureId] = (int)i;

		std::string textureFilePath = assetIdToFilePath[textureId];

		std::cout << "loading texture with id: " << textureId.toString() << " and file path: " << textureFilePath << std::endl;		

		int width, height, numChannels;
		unsigned char* raw = stbi_load(textureFilePath.c_str(), &width, &height, &numChannels, 0);

		if(raw != NULL){
			int size = width * height * numChannels;

			std::cout << "size: " << size << " width: " << width << " height: " << height << " num channels: " << numChannels << std::endl;

			std::vector<unsigned char> data;
			data.resize(size);

			for(unsigned int j = 0; j < data.size(); j++){ data[j] = raw[j]; }

			stbi_image_free(raw);

			TextureFormat format;
			switch(numChannels)
			{
				case 1:
					format = TextureFormat::Depth;
					break;
				case 2:
					format = TextureFormat::RG;
					break;
				case 3:
					format = TextureFormat::RGB;
					break;
				case 4:
					format = TextureFormat::RGBA;
					break;
				default:
					std::cout << "Error: Unsupported number of channels (" << numChannels << ") found when loading texture " << textureFilePath << " (" << textureId.toString() << ")" << std::endl;
					return false;
			}

			textures->get(i)->assetId = textureId;
			textures->get(i)->setRawTextureData(data, width, height, format);
			textures->increment();
		}
		else{
			std::cout << "Error: stbi_load failed to load texture " << textureFilePath << " (" << textureId.toString() << ") with reported reason: " << stbi_failure_reason() << std::endl;
			return false;
		}
	}

	for(unsigned int i = 0; i < shaderIds.size(); i++){
		Guid shaderId = shaderIds[i];

		assetIdToGlobalIndex[shaderId] = i;

		std::string shaderFilePath = assetIdToFilePath[shaderId];

		std::cout << "loading shader with id: " << shaderId.toString() << " and file path: " << shaderFilePath << std::endl;

		std::ifstream in(shaderFilePath.c_str());
	    std::ostringstream contents; contents << in.rdbuf(); in.close();

	    std::string shader = contents.str();

	    std::string vertexTag = "VERTEX:";
	    std::string geometryTag = "GEOMETRY:";
	    std::string fragmentTag = "FRAGMENT:";

	    size_t startOfVertexTag = shader.find(vertexTag, 0);
	    size_t startOfGeometryTag = shader.find(geometryTag, 0);
	    size_t startOfFragmentTag = shader.find(fragmentTag, 0);

	    if(startOfVertexTag == std::string::npos || startOfFragmentTag == std::string::npos){
	    	std::cout << "Error: Shader must contain both a vertex shader and a fragment shader" << std::endl;
	    	return false;
	    }

	    std::string vertexShader, geometryShader, fragmentShader;
	
	    if(startOfGeometryTag == std::string::npos){
	    	vertexShader = shader.substr(startOfVertexTag + vertexTag.length(), startOfFragmentTag - vertexTag.length());
	    	geometryShader = "";
	    	fragmentShader = shader.substr(startOfFragmentTag + fragmentTag.length(), shader.length());
	    }
	    else{
	    	vertexShader = shader.substr(startOfVertexTag + vertexTag.length(), startOfGeometryTag - vertexTag.length());
	    	geometryShader = shader.substr(startOfGeometryTag + geometryTag.length(), startOfFragmentTag - geometryTag.length());
	    	fragmentShader = shader.substr(startOfFragmentTag + fragmentTag.length(), shader.length());
	    }

	    // trim left
	    size_t firstNotOfIndex;
	    firstNotOfIndex = vertexShader.find_first_not_of("\n");
	    if(firstNotOfIndex != std::string::npos){
	    	vertexShader = vertexShader.substr(firstNotOfIndex);
	    }

	    firstNotOfIndex = geometryShader.find_first_not_of("\n");
	    if(firstNotOfIndex != std::string::npos){
	    	geometryShader = geometryShader.substr(firstNotOfIndex);
	    }

	    firstNotOfIndex = fragmentShader.find_first_not_of("\n");
	    if(firstNotOfIndex != std::string::npos){
	    	fragmentShader = fragmentShader.substr(firstNotOfIndex);
	    }

	    // trim right
	    size_t lastNotOfIndex;
	    lastNotOfIndex = vertexShader.find_last_not_of("\n");
	    if(lastNotOfIndex != std::string::npos){
	    	vertexShader.erase(lastNotOfIndex + 1);
	    }

	    lastNotOfIndex = geometryShader.find_last_not_of("\n");
	    if(lastNotOfIndex != std::string::npos){
	    	geometryShader.erase(lastNotOfIndex + 1);
	    }

	    lastNotOfIndex = fragmentShader.find_last_not_of("\n");
	    if(lastNotOfIndex != std::string::npos){
	    	fragmentShader.erase(lastNotOfIndex + 1);
	    }

	    shaders->get(i)->assetId = shaderId;
	    shaders->get(i)->vertexShader = vertexShader;
	    shaders->get(i)->geometryShader = geometryShader;
	    shaders->get(i)->fragmentShader = fragmentShader;

	    shaders->increment();

	    //std::cout << vertexShader << std::endl;
	    //std::cout << fragmentShader << std::endl;
	}

	// find all unique meshes
	std::vector<Guid> meshIds;
	for(int i = 0; i < meshRenderers->getIndex(); i++){
		bool meshIdFound = false;
		for(unsigned int j = 0; j < meshIds.size(); j++){
			if(meshRenderers->get(i)->meshId == meshIds[j]){
				meshIdFound = true;
				break;
			}
		}

		if(!meshIdFound){
			meshIds.push_back(meshRenderers->get(i)->meshId);
		}
	}

	// de-serialize all unique meshes found
	for(unsigned int i = 0; i < meshIds.size(); i++){
		Guid meshId = meshIds[i];

		assetIdToGlobalIndex[meshId] = i;

		std::string meshFilePath = assetIdToFilePath[meshId];

		std::cout << "mesh file path: " << meshFilePath << std::endl;

		MeshHeader header = {};

		FILE* file = fopen(meshFilePath.c_str(), "rb");
		size_t bytesRead;
		if (file){
			bytesRead = fread(&header, sizeof(MeshHeader), 1, file);

			meshes->get(i)->assetId = header.meshId;

			meshes->get(i)->vertices.resize(header.verticesSize);
			meshes->get(i)->normals.resize(header.normalsSize);
			meshes->get(i)->texCoords.resize(header.texCoordsSize);
			
			bytesRead += fread(&(meshes->get(i)->vertices[0]), header.verticesSize*sizeof(float), 1, file);
			bytesRead += fread(&(meshes->get(i)->normals[0]), header.normalsSize*sizeof(float), 1, file);
			bytesRead += fread(&(meshes->get(i)->texCoords[0]), header.texCoordsSize*sizeof(float), 1, file);

			meshes->increment();

			fclose(file);
		}
		else{
			std::cout << "Error: Failed to open material binary file " << meshFilePath << " for reading" << std::endl;
			return false;
		}

		std::cout << "mesh id: " << meshId.toString() << " mesh header number of vertices: " << header.verticesSize << " number of normals: " << header.normalsSize << " number of texCoords: " << header.texCoordsSize << std::endl;
	}

	return true;
}

int Manager::getNumberOfEntities()
{
	return entities->getIndex();
}

int Manager::getNumberOfSystems()
{
	return (int)systems.size();
}

Entity* Manager::getEntity(Guid id)
{
	std::map<Guid, int>::iterator it = idToGlobalIndex.find(id);
	if(it != idToGlobalIndex.end()){
		return entities->get(it->second);
	}
	else{
		std::cout << "Error: No entity with id " << id.toString() << " was found" << std::endl;
		return NULL;
	}
}

System* Manager::getSystem(Guid id)
{
	return NULL;
}

Entity* Manager::getEntityByIndex(int index)
{
	return entities->get(index);
}

System* Manager::getSystemByIndex(int index)
{
	return systems[index];
}

Line* Manager::getLine()
{
	return line;
}

Bounds* Manager::getWorldBounds()
{
	return bounds;
}

Octtree* Manager::getPhysicsTree()
{
	return physics;
}

void Manager::latentDestroy(Guid entityId)
{
	entityIdsMarkedForLatentDestroy.push_back(entityId);
}

void Manager::immediateDestroy(Guid entityId)
{
	// Entity* entity = getEntity(entityId);

	// if(entity == NULL){
	//  	std::cout << "Error: Could not find entity (" << entityId << ") when calling immediateDestroy" << std::endl;
	//  	return;
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

bool Manager::isMarkedForLatentDestroy(Guid entityId)
{
	for(unsigned int i = 0; i < entityIdsMarkedForLatentDestroy.size(); i++){
		if(entityIdsMarkedForLatentDestroy[i] == entityId){
			return true;
		}
	}

	return false;
}

std::vector<Guid> Manager::getEntitiesMarkedForLatentDestroy()
{
	return entityIdsMarkedForLatentDestroy;
}

Entity* Manager::instantiate()
{
	if(entities->getIndex() == (int)settings.maxAllowedEntities){
		std::cout << "Error: Cannot instantiate entity because we are at the settings maximum of " << settings.maxAllowedEntities << std::endl;
		return NULL;
	}

	std::cout << "number of entities: " << entities->getIndex() << std::endl;

	int index = entities->getIndex();

	entities->increment();

	std::cout << "number of entities: " << entities->getIndex() << std::endl;

	Entity* entity = entities->get(index);
	entity->entityId = Guid::newGuid();

	if(idToGlobalIndex.count(entity->entityId) == 0){
		idToGlobalIndex[entity->entityId] = index;
	}
	else{
		std::cout << "Error: Duplicate guid generated by Guid::newGuid???" << std::endl;
		return NULL;
	}

	if(entityIdToComponentIds.count(entity->entityId) == 0){
		entityIdToComponentIds[entity->entityId] = std::vector<Guid>();
	}
	else
	{
		std::cout << "Error: Duplicate guid generated by Guid::newGuid???" << std::endl;
		return NULL;
	}

	entity->setManager(this);

	return entity;
}

Entity* Manager::instantiate(Guid entityId)
{
	// if(numberOfEntities == maxAllowedEntities){
	// 	std::cout << "Error: Maximum number of entities reached. Cannot instantiate new entity" << std::endl;
	// 	return NULL;
	// }

	// Entity* entity = getEntity(entityId);

	// if(entity == NULL){ std::cout << "Error: Could not find entity (" << entityId << ") when trying to destroy" << std::endl;}

	// numberOfEntities++;

	// Entity* newEntity = &entities[numberOfEntities - 1];

	// create new entity id and set it on new entity

	// add components from component types found on input entity

	return NULL;
}

bool Manager::raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance)
{
	Ray ray;

	ray.origin = origin;
	ray.direction = direction;

	// return physics->tempIntersect(ray) != NULL;
	return physics->intersect(ray) != NULL;
}

// begin by only implementing for spheres first and later I will add for bounds, capsules etc
bool Manager::raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance, Collider** collider)
{
	Ray ray;

	ray.origin = origin;
	ray.direction = direction;

	// Object* object = physics->tempIntersect(ray);
	Object* object = physics->intersect(ray);

	if(object != NULL){
		std::map<Guid, int>::iterator it = idToGlobalIndex.find(object->id);
		if(it != idToGlobalIndex.end()){
			int colliderIndex = it->second;
			*collider = getComponentByIndex<SphereCollider>(colliderIndex);
		}
		else{
			std::cout << "Error: component id does not correspond to a global index" << std::endl;
			return false;
		}
		
		return true;
	}

	return false;
}


























bool Manager::writeToBMP(const std::string& filepath, std::vector<unsigned char>& data, int width, int height, int numChannels)
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