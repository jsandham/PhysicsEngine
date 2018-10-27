#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_set>

#include "../../include/core/Manager.h"

#include "../../include/json/json.hpp" 
#include "../../include/stb_image/stb_image.h"

#include "../../include/systems/LoadSystem.h"
#include "../../include/systems/LoadInternalSystem.h"

using namespace json;
using namespace PhysicsEngine;

Manager::Manager()
{
	entities = NULL;
	transforms = NULL;
	rigidbodies = NULL;
	cameras = NULL;
	meshRenderers = NULL;
	lineRenderers = NULL;
	directionalLights = NULL;
	spotLights = NULL;
	pointLights = NULL;
	boxColliders = NULL;
	sphereColliders = NULL;
	capsuleColliders = NULL;

	materials = NULL;
	shaders = NULL;
	textures = NULL;
	meshes = NULL;
	gmeshes = NULL;

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

	// allocate space for all entities and components
	entities = new Entity[settings.maxAllowedEntities];
	transforms = new Transform[settings.maxAllowedTransforms];
	rigidbodies = new Rigidbody[settings.maxAllowedRigidbodies];
	cameras = new Camera[settings.maxAllowedCameras];
	meshRenderers = new MeshRenderer[settings.maxAllowedMeshRenderers];
	lineRenderers = new LineRenderer[settings.maxAllowedLineRenderers];
	directionalLights = new DirectionalLight[settings.maxAllowedDirectionalLights];
	spotLights = new SpotLight[settings.maxAllowedSpotLights];
	pointLights = new PointLight[settings.maxAllowedPointLights];
	boxColliders = new BoxCollider[settings.maxAllowedBoxColliders];
	sphereColliders = new SphereCollider[settings.maxAllowedSphereColliders];
	capsuleColliders = new CapsuleCollider[settings.maxAllowedCapsuleColliders];

	// allocate space for all assets
	materials = new Material[settings.maxAllowedMaterials];
	textures = new Texture2D[settings.maxAllowedTextures];
	shaders = new Shader[settings.maxAllowedShaders];
	meshes = new Mesh[settings.maxAllowedMeshes];
	gmeshes = new GMesh[settings.maxAllowedGMeshes];

	numberOfEntities = 0;
	numberOfTransforms = 0;
	numberOfRigidbodies = 0;
	numberOfCameras = 0;
	numberOfMeshRenderers = 0;
	numberOfLineRenderers = 0;
	numberOfDirectionalLights = 0;
	numberOfSpotLights = 0;
	numberOfPointLights = 0;
	numberOfBoxColliders = 0;
	numberOfSphereColliders = 0;
	numberOfCapsuleColliders = 0;

	numberOfSystems = 0;

	numberOfMaterials = 0;
	numberOfTextures = 0;
	numberOfShaders = 0;
	numberOfMeshes = 0;
	numberOfGMeshes = 0;


	componentIdToMemoryMap[Component::getInstanceType<Transform>()] = transforms;
	componentIdToMemoryMap[Component::getInstanceType<Rigidbody>()] = rigidbodies;
	componentIdToMemoryMap[Component::getInstanceType<Camera>()] = cameras;
	componentIdToMemoryMap[Component::getInstanceType<MeshRenderer>()] = meshRenderers;
	componentIdToMemoryMap[Component::getInstanceType<LineRenderer>()] = lineRenderers;
	componentIdToMemoryMap[Component::getInstanceType<DirectionalLight>()] = directionalLights;
	componentIdToMemoryMap[Component::getInstanceType<SpotLight>()] = spotLights;
	componentIdToMemoryMap[Component::getInstanceType<PointLight>()] = pointLights;
	componentIdToMemoryMap[Component::getInstanceType<BoxCollider>()] = boxColliders;
	componentIdToMemoryMap[Component::getInstanceType<SphereCollider>()] = sphereColliders;
	componentIdToMemoryMap[Component::getInstanceType<CapsuleCollider>()] = capsuleColliders;
}

Manager::~Manager()
{
	delete [] entities;
	delete [] transforms;
	delete [] rigidbodies;
	delete [] cameras;
	delete [] meshRenderers;
	delete [] lineRenderers;
	delete [] directionalLights;
	delete [] spotLights;
	delete [] pointLights;
	delete [] boxColliders;
	delete [] sphereColliders;
	delete [] capsuleColliders;

	for(unsigned int i = 0; i < systems.size(); i++){
		delete systems[i];
	}

	delete [] materials;
	delete [] shaders;
	delete [] textures;
	delete [] meshes;
	delete [] gmeshes;
}

bool Manager::validate(std::vector<Scene> scenes, std::vector<Asset> assets)
{
	std::vector<int> materialShaderIds;
	std::vector<int> materialTextureIds;

	// check that all asset id's are unique and that the shader and texture ids on materials match actual shaders and textures
	for(unsigned int i = 0; i < assets.size(); i++){
		std::string jsonAssetFilePath = assets[i].filepath.substr(0, assets[i].filepath.find_last_of(".")) + ".json";
		std::ifstream in(jsonAssetFilePath, std::ios::in | std::ios::binary);
		std::ostringstream contents; contents << in.rdbuf(); in.close();

		json::JSON jsonAsset = JSON::Load(contents.str());

		int assetId = jsonAsset["id"].ToInt();

		std::cout << "asset id: " << assetId << " file path: " << assets[i].filepath << std::endl;
		
		if(assetIdToFilePathMap.count(assetId) == 0){
			assetIdToFilePathMap[assetId] = assets[i].filepath;
		}
		else{
			std::cout << "Error: Duplicate asset id (" << assetId << ") exists" << std::endl;
			return false;
		}

		if(assets[i].filepath.substr(assets[i].filepath.find_last_of(".") + 1) == "mat"){
			std::cout << "WTF" << std::endl;
			int shaderId = jsonAsset["shader"].ToInt();
			int mainTextureId = jsonAsset["mainTexture"].ToInt();
			int normalMapId = jsonAsset["normalMap"].ToInt();
			int specularMapId = jsonAsset["specularMap"].ToInt();

			if(shaderId != -1) { materialShaderIds.push_back(shaderId); }
			if(mainTextureId != -1) { materialTextureIds.push_back(mainTextureId); }
			if(normalMapId != -1) { materialTextureIds.push_back(normalMapId); }
			if(specularMapId != -1) { materialTextureIds.push_back(specularMapId); }
		}
	}

	for(unsigned int i = 0; i < materialShaderIds.size(); i++){
		std::map<int, std::string>::iterator it = assetIdToFilePathMap.find(materialShaderIds[i]);
		std::cout << "shader id: " << materialShaderIds[i] << std::endl;
		if(it != assetIdToFilePathMap.end()){
			std::string filepath = it->second;
			std::cout << "shader path: " << filepath << std::endl;
			if(filepath.substr(filepath.find_last_of(".") + 1) != "shader"){
				std::cout << "Error: Shader id found inside material does not correspond to a shader" << std::endl;
				return false;
			}
		}
		else{
			std::cout << "Error: Shader id in material does not match any asset file path" << std::endl;
			return false;
		}
	}

	for(unsigned int i = 0; i < materialTextureIds.size(); i++){
		std::cout << "texture id: " << materialTextureIds[i] << std::endl;
		std::map<int, std::string>::iterator it = assetIdToFilePathMap.find(materialTextureIds[i]);
		if(it != assetIdToFilePathMap.end()){
			std::string filepath = it->second;
			std::cout << "texture: " << filepath << std::endl;
			if(filepath.substr(filepath.find_last_of(".") + 1) != "png"){
				std::cout << "Error: Texture id found inside material does not correspond to a texture" << std::endl;
				return false;
			}
		}
		else{
			std::cout << "Error: Texture id in material does not match any asset file path" << std::endl;
			return false;
		}
	}

	// check that all entities and components have unique ids accross all scenes and that all scene ids are unique
	std::unordered_set<int> sceneIds;
	std::unordered_set<int> entityIds;
	std::map<int, int> componentIdToEntityIdMap;
	for(unsigned int i = 0; i < scenes.size(); i++){
		std::cout << "validating scene: " << scenes[i].name << std::endl;

		std::string jsonSceneFilePath = scenes[i].filepath.substr(0, scenes[i].filepath.find_last_of(".")) + ".json";
		std::ifstream in(jsonSceneFilePath, std::ios::in | std::ios::binary);
		std::ostringstream contents; contents << in.rdbuf(); in.close();

		json::JSON jsonScene = JSON::Load(contents.str());

		json::JSON::JSONWrapper<map<string,JSON>> objects = jsonScene.ObjectRange();
		map<string,JSON>::iterator it;

		for(it = objects.begin(); it != objects.end(); it++){
			if(it->first == "id"){ 
				int sceneId = it->second.ToInt();
				if(sceneIds.find(sceneId) == sceneIds.end()){
					sceneIds.insert(sceneId);
					continue;
				} 
				else{
					std::cout << "Error: Duplicate scene id found" << std::endl;
					return false;
				}
			}

			int objectId = std::stoi(it->first);
			std::string type = it->second["type"].ToString();

			if(type == "Entity"){
				if (entityIds.find(objectId) == entityIds.end()) {
					entityIds.insert(objectId);
				}
				else{
					std::cout << "Error: Duplicate entity id found" << std::endl;
					return false;
				}
			}
			// else if(type == "MeshRenderer"){
			// 	int entityId = it->second["entity"].ToInt();
			// 	if(componentIdToEntityIdMap.count(objectId) == 0){
			// 		componentIdToEntityIdMap[objectId] = entityId;
			// 	}
			// 	else{
			// 		std::cout << "Error: Duplicate component ids exist" << std::endl;
			// 		return false;
			// 	}
				
			// 	int meshId = it->second["mesh"].ToInt();
			// 	int materialId = it->second["material"].ToInt();

			// 	if(assetIdToFilePathMap.count(meshId) != 1){
			// 		std::cout << "Error: Mesh id (" << meshId << ") found on MeshRenderer does not match a mesh" << std::endl;
			// 		return false;
			// 	}

			// 	if(assetIdToFilePathMap.count(materialId) != 1){
			// 		std::cout << "Error: Material id (" << materialId << ") found on MeshRenderer does not match a material" << std::endl;
			// 		return false;
			// 	}
			// }


			else if(type == "PhysicsSystem"){
				std::cout << "physics system found" << std::endl;
			}
			else if(type == "RenderSystem"){
				std::cout << "render system found" << std::endl;
			}
			else if(type == "LogicSystem"){
				std::cout << "logic system found" << std::endl;
			}
			else if(type == "PlayerSystem"){
				std::cout << "player system found" << std::endl;
			}
			else{
				int entityId = it->second["entity"].ToInt();
				if(componentIdToEntityIdMap.count(objectId) == 0){
					componentIdToEntityIdMap[objectId] = entityId;
				}
				else{
					std::cout << "Error: Duplicate component ids exist" << std::endl;
					return false;
				}

				entityIdToComponentIds[entityId].push_back(objectId);

				if(type == "MeshRenderer"){
					int meshId = it->second["mesh"].ToInt();
					int materialId = it->second["material"].ToInt();

					if(assetIdToFilePathMap.count(meshId) != 1){
						std::cout << "Error: Mesh id (" << meshId << ") found on MeshRenderer does not match a mesh" << std::endl;
						return false;
					}

					if(assetIdToFilePathMap.count(materialId) != 1){
						std::cout << "Error: Material id (" << materialId << ") found on MeshRenderer does not match a material" << std::endl;
						return false;
					}
				}
			}
		}

		// check that every components entity id matches an existing entity
		std::map<int, int>::iterator iter;
		for(iter = componentIdToEntityIdMap.begin(); iter != componentIdToEntityIdMap.end(); iter++){
			int entityId = iter->second;
			if(entityIds.find(entityId) == entityIds.end()){
				std::cout << "Error: Component says it is attached to an entity that does not exist" << std::endl;
				return false;
			}
		}
	}

	return true;
}

void Manager::load(Scene scene, std::vector<Asset> assets)
{
	std::cout << "scene file path: " << scene.filepath << std::endl;

	std::string binarySceneFilePath = scene.filepath.substr(0, scene.filepath.find_last_of(".")) + ".scene";

	std::cout << "scene file path: " << scene.filepath << " binary scene file path: " << binarySceneFilePath << " size of camera: " << sizeof(Camera) << std::endl;

	SceneHeader sceneHeader = {};
	FILE* file = fopen(binarySceneFilePath.c_str(), "rb");
	size_t bytesRead;
	if (file){
		bytesRead = fread(&sceneHeader, sizeof(SceneHeader), 1, file);

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

		int existingNumberOfEntities = numberOfEntities;
		int existingNumberOfTransforms = numberOfTransforms;
		int existingNumberOfRigidbodies = numberOfRigidbodies;
		int existingNumberOfCameras = numberOfCameras;
		int existingNumberOfMeshRenderers = numberOfMeshRenderers;
		int existingNumberOfLineRenderers = numberOfLineRenderers;
		int existingNumberOfDirectionalLights = numberOfDirectionalLights;
		int existingNumberOfSpotLights = numberOfSpotLights;
		int existingNumberOfPointLights = numberOfPointLights;
		int existingNumberOfBoxColliders = numberOfBoxColliders;
		int existingNumberOfSphereColliders = numberOfSphereColliders;
		int existingNumberOfCapsuleColliders = numberOfCapsuleColliders;

		numberOfEntities = existingNumberOfEntities + sceneHeader.numberOfEntities;
		numberOfTransforms = existingNumberOfTransforms + sceneHeader.numberOfTransforms;
		numberOfRigidbodies = existingNumberOfRigidbodies + sceneHeader.numberOfRigidbodies;
		numberOfCameras = existingNumberOfCameras + sceneHeader.numberOfCameras;
		numberOfMeshRenderers = existingNumberOfMeshRenderers + sceneHeader.numberOfMeshRenderers;
		numberOfLineRenderers = existingNumberOfLineRenderers + sceneHeader.numberOfLineRenderers;
		numberOfDirectionalLights = existingNumberOfDirectionalLights + sceneHeader.numberOfDirectionalLights;
		numberOfSpotLights = existingNumberOfSpotLights + sceneHeader.numberOfSpotLights;
		numberOfPointLights = existingNumberOfPointLights + sceneHeader.numberOfPointLights;
		numberOfBoxColliders = existingNumberOfBoxColliders + sceneHeader.numberOfBoxColliders;
		numberOfSphereColliders = existingNumberOfSphereColliders + sceneHeader.numberOfSphereColliders;
		numberOfCapsuleColliders = existingNumberOfCapsuleColliders + sceneHeader.numberOfCapsuleColliders;

		std::cout << "numberOfEntities: " << numberOfEntities << std::endl;
		std::cout << "numberOfTransforms: " << numberOfTransforms << std::endl;
		std::cout << "numberOfRigidbodies: " << numberOfRigidbodies << std::endl;
		std::cout << "numberOfCameras: " << numberOfCameras << std::endl;
		std::cout << "numberOfMeshRenderers: " << numberOfMeshRenderers << std::endl;
		std::cout << "numberOfLineRenderers: " << numberOfLineRenderers << std::endl;
		std::cout << "numberOfDirectionalLights: " << numberOfDirectionalLights << std::endl;
		std::cout << "numberOfSpotLights: " << numberOfSpotLights << std::endl;
		std::cout << "numberOfPointLights: " << numberOfPointLights << std::endl;
		std::cout << "numberOfBoxColliders: " << numberOfBoxColliders << std::endl;
		std::cout << "numberOfSphereColliders: " << numberOfSphereColliders << std::endl;
		std::cout << "numberOfCapsuleColliders: " << numberOfCapsuleColliders << std::endl;
		std::cout << "numberOfSystems: " << numberOfSystems << std::endl;

		bool error = numberOfEntities > settings.maxAllowedEntities;
		error |= numberOfTransforms > settings.maxAllowedTransforms;
		error |= numberOfRigidbodies > settings.maxAllowedRigidbodies;
		error |= numberOfCameras > settings.maxAllowedCameras;
		error |= numberOfMeshRenderers > settings.maxAllowedMeshRenderers;
		error |= numberOfLineRenderers > settings.maxAllowedLineRenderers;
		error |= numberOfDirectionalLights > settings.maxAllowedDirectionalLights;
		error |= numberOfSpotLights > settings.maxAllowedSpotLights;
		error |= numberOfPointLights > settings.maxAllowedPointLights;
		error |= numberOfBoxColliders > settings.maxAllowedBoxColliders;
		error |= numberOfSphereColliders > settings.maxAllowedSphereColliders;
		error |= numberOfCapsuleColliders > settings.maxAllowedCapsuleColliders;

		if(error){
			std::cout << "Error: Number of entities, components, or systems exceeds maximum allowed. Please increase max allowed in scene settings." << std::endl;
			return;
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
			return;
		}

		// de-serialize entities and components
		bytesRead = fread(&entities[existingNumberOfEntities], sceneHeader.numberOfEntities*sizeof(Entity), 1, file);
		bytesRead = fread(&transforms[existingNumberOfTransforms], sceneHeader.numberOfTransforms*sizeof(Transform), 1, file);
		bytesRead = fread(&rigidbodies[existingNumberOfRigidbodies], sceneHeader.numberOfRigidbodies*sizeof(Rigidbody), 1, file);
		bytesRead = fread(&cameras[existingNumberOfCameras], sceneHeader.numberOfCameras*sizeof(Camera), 1, file);
		bytesRead = fread(&meshRenderers[existingNumberOfMeshRenderers], sceneHeader.numberOfMeshRenderers*sizeof(MeshRenderer), 1, file);
		bytesRead = fread(&lineRenderers[existingNumberOfLineRenderers], sceneHeader.numberOfLineRenderers*sizeof(LineRenderer), 1, file);
		bytesRead = fread(&directionalLights[existingNumberOfDirectionalLights], sceneHeader.numberOfDirectionalLights*sizeof(DirectionalLight), 1, file);
		bytesRead = fread(&spotLights[existingNumberOfSpotLights], sceneHeader.numberOfSpotLights*sizeof(SpotLight), 1, file);
		bytesRead = fread(&pointLights[existingNumberOfPointLights], sceneHeader.numberOfPointLights*sizeof(PointLight), 1, file);
		bytesRead = fread(&boxColliders[existingNumberOfBoxColliders], sceneHeader.numberOfBoxColliders*sizeof(BoxCollider), 1, file);
		bytesRead = fread(&sphereColliders[existingNumberOfSphereColliders], sceneHeader.numberOfSphereColliders*sizeof(SphereCollider), 1, file);
		bytesRead = fread(&capsuleColliders[existingNumberOfCapsuleColliders], sceneHeader.numberOfCapsuleColliders*sizeof(CapsuleCollider), 1, file);

		// what if I wrote the size of the system in the binary file, followed by the actual system byte data?
		// for(int i = 0; i < numberOfSystems; i++){
		while(true){
			size_t sizeOfSystem;
			bytesRead = fread(&sizeOfSystem, sizeof(size_t), 1, file);

			std::cout << "bytes read: " << bytesRead << " size of system data read: " << sizeOfSystem << std::endl;

			if(bytesRead != 1){
				break;
			}

			unsigned char* data = new unsigned char[sizeOfSystem];
			bytesRead = fread(data, sizeOfSystem * sizeof(unsigned char), 1, file);

			std::cout << "AAAAAAAAAAAAAAAAAAAAA" << std::endl;

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

			std::cout << "BBBBBBBB" << std::endl;

			if(system == NULL){
				std::cout << "Error: Could not load system" << std::endl;
				return;
			}

			systems.push_back(system);

			delete [] data;
		}

		numberOfSystems = (int)systems.size();

		fclose(file);
	}
	else{
		std::cout << "Error: Failed to open scene binary file " << binarySceneFilePath << " for reading" << std::endl;
		return;
	}

	// set manager on entites and components
	for(int i = 0; i < numberOfEntities; i++){ entities[i].setManager(this); }
	for(int i = 0; i < numberOfTransforms; i++){ transforms[i].setManager(this); }
	for(int i = 0; i < numberOfRigidbodies; i++){ rigidbodies[i].setManager(this); }
	for(int i = 0; i < numberOfCameras; i++){ cameras[i].setManager(this); }
	for(int i = 0; i < numberOfMeshRenderers; i++){ meshRenderers[i].setManager(this); }
	for(int i = 0; i < numberOfLineRenderers; i++){ lineRenderers[i].setManager(this); }
	for(int i = 0; i < numberOfDirectionalLights; i++){ directionalLights[i].setManager(this); }
	for(int i = 0; i < numberOfSpotLights; i++){ spotLights[i].setManager(this); }
	for(int i = 0; i < numberOfPointLights; i++){ pointLights[i].setManager(this); }
	for(int i = 0; i < numberOfBoxColliders; i++){ boxColliders[i].setManager(this); }
	for(int i = 0; i < numberOfSphereColliders; i++){ sphereColliders[i].setManager(this); }
	for(int i = 0; i < numberOfCapsuleColliders; i++){ capsuleColliders[i].setManager(this); }

	// set manager on systems
	for(unsigned int i = 0; i < systems.size(); i++){
		systems[i]->setManager(this);
	}

	// map entity/component id to its global array index
	for(int i = 0; i < numberOfEntities; i++){ idToGlobalIndexMap[entities[i].entityId] = i; }
	for(int i = 0; i < numberOfTransforms; i++){ idToGlobalIndexMap[transforms[i].componentId] = i; }
	for(int i = 0; i < numberOfRigidbodies; i++){ idToGlobalIndexMap[rigidbodies[i].componentId] = i; }
	for(int i = 0; i < numberOfCameras; i++){ idToGlobalIndexMap[cameras[i].componentId] = i; }
	for(int i = 0; i < numberOfMeshRenderers; i++){ idToGlobalIndexMap[meshRenderers[i].componentId] = i; }
	for(int i = 0; i < numberOfLineRenderers; i++){ idToGlobalIndexMap[lineRenderers[i].componentId] = i; }
	for(int i = 0; i < numberOfDirectionalLights; i++){ idToGlobalIndexMap[directionalLights[i].componentId] = i; }
	for(int i = 0; i < numberOfSpotLights; i++){ idToGlobalIndexMap[spotLights[i].componentId] = i; }
	for(int i = 0; i < numberOfPointLights; i++){ idToGlobalIndexMap[pointLights[i].componentId] = i; }
	for(int i = 0; i < numberOfBoxColliders; i++){ idToGlobalIndexMap[boxColliders[i].componentId] = i; }
	for(int i = 0; i < numberOfSphereColliders; i++){ idToGlobalIndexMap[sphereColliders[i].componentId] = i; }
	for(int i = 0; i < numberOfCapsuleColliders; i++){ idToGlobalIndexMap[capsuleColliders[i].componentId] = i; }

	// map component id to its type
	for(int i = 0; i < numberOfTransforms; i++){ componentIdToTypeMap[transforms[i].componentId] = Component::getInstanceType<Transform>(); }
	for(int i = 0; i < numberOfRigidbodies; i++){ componentIdToTypeMap[rigidbodies[i].componentId] = Component::getInstanceType<Rigidbody>(); }
	for(int i = 0; i < numberOfCameras; i++){ componentIdToTypeMap[cameras[i].componentId] = Component::getInstanceType<Camera>(); }
	for(int i = 0; i < numberOfMeshRenderers; i++){ componentIdToTypeMap[meshRenderers[i].componentId] = Component::getInstanceType<MeshRenderer>(); }
	for(int i = 0; i < numberOfLineRenderers; i++){ componentIdToTypeMap[lineRenderers[i].componentId] = Component::getInstanceType<LineRenderer>(); }
	for(int i = 0; i < numberOfDirectionalLights; i++){ componentIdToTypeMap[directionalLights[i].componentId] = Component::getInstanceType<DirectionalLight>(); }
	for(int i = 0; i < numberOfSpotLights; i++){ componentIdToTypeMap[spotLights[i].componentId] = Component::getInstanceType<SpotLight>(); }
	for(int i = 0; i < numberOfPointLights; i++){ componentIdToTypeMap[pointLights[i].componentId] = Component::getInstanceType<PointLight>(); }
	for(int i = 0; i < numberOfBoxColliders; i++){ componentIdToTypeMap[boxColliders[i].componentId] = Component::getInstanceType<BoxCollider>(); }
	for(int i = 0; i < numberOfSphereColliders; i++){ componentIdToTypeMap[sphereColliders[i].componentId] = Component::getInstanceType<SphereCollider>(); }
	for(int i = 0; i < numberOfCapsuleColliders; i++){ componentIdToTypeMap[capsuleColliders[i].componentId] = Component::getInstanceType<CapsuleCollider>(); }

	std::cout << "number of mesh renderers: " << numberOfMeshRenderers << std::endl;
	for(int i = 0; i < numberOfMeshRenderers; i++){
		std::cout << "material id found on meshrenderer: " << meshRenderers[i].materialId << " mesh id: " << meshRenderers[i].meshId << std::endl;
	}

	// find all unique materials
	std::vector<int> materialIds;
	for(int i = 0; i < numberOfMeshRenderers; i++){
		bool materialIdFound = false;
		for(unsigned int j = 0; j < materialIds.size(); j++){
			if(meshRenderers[i].materialId == materialIds[j]){
				materialIdFound = true;
				break;
			}
		}

		if(!materialIdFound){
			std::cout << "material id: " << meshRenderers[i].materialId << " i: " << i << std::endl;
			materialIds.push_back(meshRenderers[i].materialId);
		}
	}

	numberOfMaterials = (int)materialIds.size();

	// de-serialize all unique materials found
	for(unsigned int i = 0; i < materialIds.size(); i++){
		int materialId = materialIds[i];

		assetIdToGlobalIndexMap[materialId] = i;

		std::string materialFilePath = assetIdToFilePathMap[materialId];

		std::cout << "material id: " << materialId << " material file path: " << materialFilePath << std::endl;

		FILE* file = fopen(materialFilePath.c_str(), "rb");
		size_t bytesRead;
		if (file){
			bytesRead = fread(&materials[i], sizeof(Material), 1, file);
			std::cout << "number of bytes read from file: " << bytesRead << std::endl;
		}
		else{
			std::cout << "Error: Failed to open material binary file " << materialFilePath << " for reading" << std::endl;
			return;
		}

		std::cout << "material id: " << materials[i].materialId << " texture id: " << materials[i].textureId << " shader id: " << materials[i].shaderId << std::endl;
	}

	// set manager on materials
	for(int i = 0; i < numberOfMaterials; i++){
		materials[i].setManager(this);
	}

	// find all unique textures and shaders 
	std::vector<int> textureIds;
	std::vector<int> shaderIds;
	for(unsigned int i = 0; i < materialIds.size(); i++){
		if(materials[i].textureId != -1){
			bool mainTextureIdFound = false;
			for(unsigned int j = 0; j < textureIds.size(); j++){
				if(materials[i].textureId == textureIds[j]){
					mainTextureIdFound = true;
					break;
				}
			}

			if(!mainTextureIdFound){
				textureIds.push_back(materials[i].textureId);
			}
		}

		if(materials[i].normalMapId != -1){
			bool normalMapIdFound = false;
			for(unsigned int j = 0; j < textureIds.size(); j++){
				if(materials[i].normalMapId == textureIds[j]){
					normalMapIdFound = true;
					break;
				}
			}

			if(!normalMapIdFound){
				textureIds.push_back(materials[i].normalMapId);
			}
		}

		if(materials[i].specularMapId != -1){
			bool specularMapIdFound = false;
			for(unsigned int j = 0; j < textureIds.size(); j++){
				if(materials[i].specularMapId == textureIds[j]){
					specularMapIdFound = true;
					break;
				}
			}

			if(!specularMapIdFound){
				textureIds.push_back(materials[i].specularMapId);
			}
		}

		bool shaderIdFound = false;
		for(unsigned int j = 0; j < shaderIds.size(); j++){
			if(materials[i].shaderId == shaderIds[j]){
				shaderIdFound = true;
				break;
			}
		}

		if(!shaderIdFound){
			shaderIds.push_back(materials[i].shaderId);
		}
	}

	// TODO: maybe store internally used shaders a static strings directly in source code?
	// include internally used depth shader
	shaderIds.push_back(34343434);

	numberOfTextures = (int)textureIds.size();
	numberOfShaders = (int)shaderIds.size();

	// de-serialize all unique textures and shaders found
	for(unsigned int i = 0; i < textureIds.size(); i++){
		int textureId = textureIds[i];

		std::cout << "textureId: " << textureId << " i: " << i << std::endl;

		assetIdToGlobalIndexMap[textureId] = (int)i;

		std::string textureFilePath = assetIdToFilePathMap[textureId];

		std::cout << "loading texture with id: " << textureId << " and file path: " << textureFilePath << std::endl;		

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
					std::cout << "Error: Unsupported number of channels (" << numChannels << ") found when loading texture " << textureFilePath << " (" << textureId << ")" << std::endl;
					return;
			}

			textures[i].textureId = textureId;
			textures[i].setRawTextureData(data, width, height, format);
		}
		else{
			std::cout << "Error: stbi_load failed to load texture " << textureFilePath << " (" << textureId << ") with reported reason: " << stbi_failure_reason() << std::endl;
			return;
		}
	}

	std::cout << "textures loaded" << std::endl;

	for(unsigned int i = 0; i < shaderIds.size(); i++){
		int shaderId = shaderIds[i];

		assetIdToGlobalIndexMap[shaderId] = i;

		std::string shaderFilePath = assetIdToFilePathMap[shaderId];

		std::cout << "loading shader with id: " << shaderId << " and file path: " << shaderFilePath << std::endl;

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
	    	return;
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

	    shaders[i].shaderId = shaderId;
	    shaders[i].vertexShader = vertexShader;
	    shaders[i].geometryShader = geometryShader;
	    shaders[i].fragmentShader = fragmentShader;

	    //std::cout << vertexShader << std::endl;
	    //std::cout << fragmentShader << std::endl;
	}

	// find all unique meshes
	std::vector<int> meshIds;
	for(int i = 0; i < numberOfMeshRenderers; i++){
		bool meshIdFound = false;
		for(unsigned int j = 0; j < meshIds.size(); j++){
			if(meshRenderers[i].meshId == meshIds[j]){
				meshIdFound = true;
				break;
			}
		}

		if(!meshIdFound){
			meshIds.push_back(meshRenderers[i].meshId);
		}
	}

	numberOfMeshes = (int)meshIds.size();

	// de-serialize all unique meshes found
	for(unsigned int i = 0; i < meshIds.size(); i++){
		int meshId = meshIds[i];

		assetIdToGlobalIndexMap[meshId] = i;

		std::string meshFilePath = assetIdToFilePathMap[meshId];

		std::cout << "mesh file path: " << meshFilePath << std::endl;

		MeshHeader header = {};

		FILE* file = fopen(meshFilePath.c_str(), "rb");
		size_t bytesRead;
		if (file){
			bytesRead = fread(&header, sizeof(MeshHeader), 1, file);

			meshes[i].meshId = header.meshId;

			meshes[i].vertices.resize(header.verticesSize);
			meshes[i].normals.resize(header.normalsSize);
			meshes[i].texCoords.resize(header.texCoordsSize);
			
			bytesRead += fread(&(meshes[i].vertices[0]), header.verticesSize*sizeof(float), 1, file);
			bytesRead += fread(&(meshes[i].normals[0]), header.normalsSize*sizeof(float), 1, file);
			bytesRead += fread(&(meshes[i].texCoords[0]), header.texCoordsSize*sizeof(float), 1, file);
			std::cout << "number of bytes read from file: " << bytesRead << std::endl;

			fclose(file);
		}
		else{
			std::cout << "Error: Failed to open material binary file " << meshFilePath << " for reading" << std::endl;
			return;
		}

		std::cout << "mesh id: " << meshId << " mesh header number of vertices: " << header.verticesSize << " number of normals: " << header.normalsSize << " number of texCoords: " << header.texCoordsSize << std::endl;
	}

	std::cout << "numberOfMeshRenderers: " << numberOfMeshRenderers << std::endl;


	std::cout << "transform type: " << Component::getInstanceType<Transform>() << std::endl;
	std::cout << "transform type: " << Component::getInstanceType<Transform>() << std::endl;
	std::cout << "rigidbody type: " << Component::getInstanceType<Rigidbody>() << std::endl;
	std::cout << "camera type: " << Component::getInstanceType<Camera>() << std::endl;
	std::cout << "mesh renderer type: " << Component::getInstanceType<MeshRenderer>() << std::endl;
	std::cout << "camera type: " << Component::getInstanceType<Camera>() << std::endl;
}

int Manager::getNumberOfEntities()
{
	return numberOfEntities;
}

int Manager::getNumberOfTransforms()
{
	return numberOfTransforms;
}

int Manager::getNumberOfRigidbodies()
{
	return numberOfRigidbodies;
}

int Manager::getNumberOfCameras()
{
	return numberOfCameras;
}

int Manager::getNumberOfMeshRenderers()
{
	return numberOfMeshRenderers;
}

int Manager::getNumberOfLineRenderers()
{
	return numberOfLineRenderers;
}

int Manager::getNumberOfDirectionalLights()
{
	return numberOfDirectionalLights;
}

int Manager::getNumberOfSpotLights()
{
	return numberOfSpotLights;
}

int Manager::getNumberOfPointLights()
{
	return numberOfPointLights;
}

int Manager::getNumberOfMaterials()
{
	return numberOfMaterials;
}

int Manager::getNumberOfShaders()
{
	return numberOfShaders;
}

int Manager::getNumberOfTextures()
{
	return numberOfTextures;
}

int Manager::getNumberOfMeshes()
{
	return numberOfMeshes;
}

int Manager::getNumberOfGmeshes()
{
	return numberOfGMeshes;
}

int Manager::getNumberOfBoxColliders()
{
	return numberOfBoxColliders;
}

int Manager::getNumberOfSphereColliders()
{
	return numberOfSphereColliders;
}

int Manager::getNumberOfCapsuleColliders()
{
	return numberOfCapsuleColliders;
}

int Manager::getNumberOfSystems()
{
	return numberOfSystems;
}

Entity* Manager::getEntity(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &entities[it->second];
	}
	else{
		std::cout << "Error: No entity with id " << id << " was found" << std::endl;
		return NULL;
	}
}

Transform* Manager::getTransform(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &transforms[it->second];
	}
	else{
		std::cout << "Error: No transform with id " << id << " was found" << std::endl;
		return NULL;
	}
}

Rigidbody* Manager::getRigidbody(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &rigidbodies[it->second];
	}
	else{
		std::cout << "Error: No rigidbody with id " << id << " was found" << std::endl;
		return NULL;
	}
}

Camera* Manager::getCamera(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &cameras[it->second];
	}
	else{
		std::cout << "Error: No camera with id " << id << " was found" << std::endl;
		return NULL;
	}
}

MeshRenderer* Manager::getMeshRenderer(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &meshRenderers[it->second];
	}
	else{
		std::cout << "Error: No entity with id " << id << " was found" << std::endl;
		return NULL;
	}
}

LineRenderer* Manager::getLineRenderer(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &lineRenderers[it->second];
	}
	else{
		std::cout << "Error: No entity with id " << id << " was found" << std::endl;
		return NULL;
	}
}

DirectionalLight* Manager::getDirectionalLight(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &directionalLights[it->second];
	}
	else{
		std::cout << "Error: No mesh renderer with id " << id << " was found" << std::endl;
		return NULL;
	}
}

SpotLight* Manager::getSpotLight(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &spotLights[it->second];
	}
	else{
		std::cout << "Error: No spot light with id " << id << " was found" << std::endl;
		return NULL;
	}
}	

PointLight* Manager::getPointLight(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &pointLights[it->second];
	}
	else{
		std::cout << "Error: No point light with id " << id << " was found" << std::endl;
		return NULL;
	}
}

BoxCollider* Manager::getBoxCollider(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &boxColliders[it->second];
	}
	else{
		std::cout << "Error: No box collider with id " << id << " was found" << std::endl;
		return NULL;
	}
}

SphereCollider* Manager::getSphereCollider(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &sphereColliders[it->second];
	}
	else{
		std::cout << "Error: No sphere colliders with id " << id << " was found" << std::endl;
		return NULL;
	}
}

CapsuleCollider* Manager::getCapsuleCollider(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &capsuleColliders[it->second];
	}
	else{
		std::cout << "Error: No capsule colliders with id " << id << " was found" << std::endl;
		return NULL;
	}
}

System* Manager::getSystem(int id)
{
	return NULL;
}

Material* Manager::getMaterial(int id)
{
	std::map<int, int>::iterator it = assetIdToGlobalIndexMap.find(id);
	if(it != assetIdToGlobalIndexMap.end()){
		return &materials[it->second];
	}
	else{
		std::cout << "Error: No material with id " << id << " was found" << std::endl;
		return NULL;
	}
}

Shader* Manager::getShader(int id)
{
	std::map<int, int>::iterator it = assetIdToGlobalIndexMap.find(id);
	if(it != assetIdToGlobalIndexMap.end()){
		return &shaders[it->second];
	}
	else{
		std::cout << "Error: No shader with id " << id << " was found" << std::endl;
		return NULL;
	}
}

Texture2D* Manager::getTexture2D(int id)
{
	std::map<int, int>::iterator it = assetIdToGlobalIndexMap.find(id);
	if(it != assetIdToGlobalIndexMap.end()){
		return &textures[it->second];
	}
	else{
		//std::cout << "Error: No texture with id " << id << " was found" << std::endl;
		return NULL;
	}
}

Mesh* Manager::getMesh(int id)
{
	std::map<int, int>::iterator it = assetIdToGlobalIndexMap.find(id);
	if(it != assetIdToGlobalIndexMap.end()){
		return &meshes[it->second];
	}
	else{
		std::cout << "Error: No mesh with id " << id << " was found" << std::endl;
		return NULL;
	}
}

GMesh* Manager::getGMesh(int id)
{
	std::map<int, int>::iterator it = assetIdToGlobalIndexMap.find(id);
	if(it != assetIdToGlobalIndexMap.end()){
		return &gmeshes[it->second];
	}
	else{
		std::cout << "Error: No gmesh with id " << id << " was found" << std::endl;
		return NULL;
	}
}

Entity* Manager::getEntityByIndex(int index)
{
	return &entities[index];
}

Transform* Manager::getTransformByIndex(int index)
{
	return &transforms[index];
}

Rigidbody* Manager::getRigidbodyByIndex(int index)
{
	return &rigidbodies[index];
}

Camera* Manager::getCameraByIndex(int index)
{
	return &cameras[index];
}

MeshRenderer* Manager::getMeshRendererByIndex(int index)
{
	return &meshRenderers[index];
}

LineRenderer* Manager::getLineRendererByIndex(int index)
{
	return &lineRenderers[index];
}

DirectionalLight* Manager::getDirectionalLightByIndex(int index)
{
	return &directionalLights[index];
}

SpotLight* Manager::getSpotLightByIndex(int index)
{
	return &spotLights[index];
}

PointLight* Manager::getPointLightByIndex(int index)
{
	return &pointLights[index];
}

BoxCollider* Manager::getBoxColliderByIndex(int index)
{
	return &boxColliders[index];
}

SphereCollider* Manager::getSphereColliderByIndex(int index)
{
	return &sphereColliders[index];
}

CapsuleCollider* Manager::getCapsuleColliderByIndex(int index)
{
	return &capsuleColliders[index];
}

System* Manager::getSystemByIndex(int index)
{
	return systems[index];
}

Material* Manager::getMaterialByIndex(int index)
{
	return &materials[index];
}

Shader* Manager::getShaderByIndex(int index)
{
	return &shaders[index];
}

Texture2D* Manager::getTexture2DByIndex(int index)
{
	return &textures[index];
}

Mesh* Manager::getMeshByIndex(int index)
{
	return &meshes[index];
}

GMesh* Manager::getGMeshByIndex(int index)
{
	return &gmeshes[index];
}

void Manager::latentDestroy(int entityId)
{
	entitiesMarkedForLatentDestroy.push_back(entityId);
}

void Manager::immediateDestroy(int entityId)
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

bool Manager::isMarkedForLatentDestroy(int entityId)
{
	for(unsigned int i = 0; i < entitiesMarkedForLatentDestroy.size(); i++){
		if(entitiesMarkedForLatentDestroy[i] == entityId){
			return true;
		}
	}

	return false;
}

std::vector<int> Manager::getEntitiesMarkedForLatentDestroy()
{
	return entitiesMarkedForLatentDestroy;
}

Entity* Manager::instantiate()
{
	if(numberOfEntities == settings.maxAllowedEntities){
		std::cout << "Error: Cannot instantiate entity because we are at the settings maximum of " << settings.maxAllowedEntities << std::endl;
		return NULL;
	}

	numberOfEntities++;

	Entity* entity = &entities[numberOfEntities - 1];

	// TODO: set entity id on newly created entity and insert into idToGlobalIndexMap and entityIdToComponentIds maps

	return entity;
}

Entity* Manager::instantiate(int entityId)
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