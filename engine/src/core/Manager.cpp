#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_set>

#include "../../include/core/Manager.h"

#include "../../include/json/json.hpp" 
#include "../../include/stb_image/stb_image.h"

using namespace json;
using namespace PhysicsEngine;

Manager::Manager()
{
	entities = NULL;
	transforms = NULL;
	rigidbodies = NULL;
	cameras = NULL;
	meshRenderers = NULL;
	directionalLights = NULL;
	spotLights = NULL;
	pointLights = NULL;

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
			settings.maxAllowedDirectionalLights = it->second["maxAllowedDirectionalLights"].ToInt();
			settings.maxAllowedSpotLights = it->second["maxAllowedSpotLights"].ToInt();
			settings.maxAllowedPointLights = it->second["maxAllowedPointLights"].ToInt();

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
	error |= settings.maxAllowedDirectionalLights <= 0;
	error |= settings.maxAllowedSpotLights <= 0;
	error |= settings.maxAllowedPointLights <= 0;

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
	directionalLights = new DirectionalLight[settings.maxAllowedDirectionalLights];
	spotLights = new SpotLight[settings.maxAllowedSpotLights];
	pointLights = new PointLight[settings.maxAllowedPointLights];

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
	numberOfDirectionalLights = 0;
	numberOfSpotLights = 0;
	numberOfPointLights = 0;

	numberOfMaterials = 0;
	numberOfTextures = 0;
	numberOfShaders = 0;
	numberOfMeshes = 0;
	numberOfGMeshes = 0;
}

Manager::~Manager()
{
	delete [] entities;
	delete [] transforms;
	delete [] rigidbodies;
	delete [] cameras;
	delete [] meshRenderers;
	delete [] directionalLights;
	delete [] spotLights;
	delete [] pointLights;

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

		if(assets[i].filepath.substr(assets[i].filepath.find_last_of(".") + 1) == "material"){
			materialShaderIds.push_back(jsonAsset["shaderId"].ToInt());
			materialTextureIds.push_back(jsonAsset["textureId"].ToInt());
		}
	}

	for(unsigned int i = 0; i < materialShaderIds.size(); i++){
		std::map<int, std::string>::iterator it = assetIdToFilePathMap.find(materialShaderIds[i]);
		if(it != assetIdToFilePathMap.end()){
			std::string filepath = it->second;
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
		std::map<int, std::string>::iterator it = assetIdToFilePathMap.find(materialTextureIds[i]);
		if(it != assetIdToFilePathMap.end()){
			std::string filepath = it->second;
			if(filepath.substr(filepath.find_last_of(".") + 1) != "texture"){
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
			else{
				int entityId = it->second["entity"].ToInt();
				if(componentIdToEntityIdMap.count(objectId) == 0){
					componentIdToEntityIdMap[objectId] = entityId;
				}
				else{
					std::cout << "Error: Duplicate component ids exist" << std::endl;
					return false;
				}

				if(type == "MeshRenderer"){
					int meshId = it->second["mesh"].ToInt();
					int materialId = it->second["material"].ToInt();

					if(assetIdToFilePathMap.count(meshId) != 1){
						std::cout << "Error: Mesh id found on MeshRenderer does not match a mesh" << std::endl;
						return false;
					}

					if(assetIdToFilePathMap.count(materialId) != 1){
						std::cout << "Error: Material id found on MeshRenderer does not match a material" << std::endl;
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
	// create asset id to filepath map
	// for(unsigned int i = 0; i < assets.size(); i++){
	// 	// open asset files json object and get asset id
	// 	std::string jsonAssetFilePath = assets[i].filepath.substr(0, assets[i].filepath.find_last_of(".")) + ".json";
	// 	std::cout << "jsonAssetFilePath: " << jsonAssetFilePath << std::endl;
	// 	std::ifstream in(jsonAssetFilePath, std::ios::in | std::ios::binary);
	// 	std::ostringstream contents; contents << in.rdbuf(); in.close();

	// 	json::JSON jsonAsset = JSON::Load(contents.str());

	// 	int assetId = jsonAsset["id"].ToInt();

	// 	std::cout << "asset id: " << assetId << " file path: " << assets[i].filepath << std::endl;
		
	// 	assetIdToFilePathMap[assetId] = assets[i].filepath;
	// }

	//for(unsigned int i = 0; i < numberOfEntities; i++){
	//	if(!entities[i].isActive){
	//
	//	}
	//}

	


	std::cout << "scene file path: " << scene.filepath << std::endl;

	std::string binarySceneFilePath = scene.filepath.substr(0, scene.filepath.find_last_of(".")) + ".scene";

	std::cout << "scene file path: " << scene.filepath << " binary scene file path: " << binarySceneFilePath << " size of camera: " << sizeof(Camera) << std::endl;

	SceneHeader sceneHeader = {};
	FILE* file = fopen(binarySceneFilePath.c_str(), "rb");
	size_t bytesRead;
	if (file){
		bytesRead = fread(&sceneHeader, sizeof(SceneHeader), 1, file);

		std::cout << "de-serialized scene header file contains the following information: " << std::endl;

		std::cout << "sizeOfEntity: " << sceneHeader.sizeOfEntity << std::endl;
		std::cout << "sizeOfTransform: " << sceneHeader.sizeOfTransform << std::endl;
		std::cout << "sizeOfRigidbodies: " << sceneHeader.sizeOfRigidbody << std::endl;
		std::cout << "sizeOfCameras: " << sceneHeader.sizeOfCamera << std::endl;
		std::cout << "sizeOfMeshRenderer: " << sceneHeader.sizeOfMeshRenderer << std::endl;
		std::cout << "sizeOfDirectionalLight: " << sceneHeader.sizeOfDirectionalLight << std::endl;
		std::cout << "sizeOfSpotLight: " << sceneHeader.sizeOfSpotLight << std::endl;
		std::cout << "sizeOfPointLight: " << sceneHeader.sizeOfPointLight << std::endl;

		int existingNumberOfEntities = numberOfEntities;
		int existingNumberOfTransforms = numberOfTransforms;
		int existingNumberOfRigidbodies = numberOfRigidbodies;
		int existingNumberOfCameras = numberOfCameras;
		int existingNumberOfMeshRenderers = numberOfMeshRenderers;
		int existingNumberOfDirectionalLights = numberOfDirectionalLights;
		int existingNumberOfSpotLights = numberOfSpotLights;
		int existingNumberOfPointLights = numberOfPointLights;

		numberOfEntities = existingNumberOfEntities + sceneHeader.numberOfEntities;
		numberOfTransforms = existingNumberOfTransforms + sceneHeader.numberOfTransforms;
		numberOfRigidbodies = existingNumberOfRigidbodies + sceneHeader.numberOfRigidbodies;
		numberOfCameras = existingNumberOfCameras + sceneHeader.numberOfCameras;
		numberOfMeshRenderers = existingNumberOfMeshRenderers + sceneHeader.numberOfMeshRenderers;
		numberOfDirectionalLights = existingNumberOfDirectionalLights + sceneHeader.numberOfDirectionalLights;
		numberOfSpotLights = existingNumberOfSpotLights + sceneHeader.numberOfSpotLights;
		numberOfPointLights = existingNumberOfPointLights + sceneHeader.numberOfPointLights;

		std::cout << "numberOfEntities: " << numberOfEntities << std::endl;
		std::cout << "numberOfTransforms: " << numberOfTransforms << std::endl;
		std::cout << "numberOfRigidbodies: " << numberOfRigidbodies << std::endl;
		std::cout << "numberOfCameras: " << numberOfCameras << std::endl;
		std::cout << "numberOfMeshRenderers: " << numberOfMeshRenderers << std::endl;
		std::cout << "numberOfDirectionalLights: " << numberOfDirectionalLights << std::endl;
		std::cout << "numberOfSpotLights: " << numberOfSpotLights << std::endl;
		std::cout << "numberOfPointLights: " << numberOfPointLights << std::endl;

		bool error = numberOfEntities > settings.maxAllowedEntities;
		error |= numberOfTransforms > settings.maxAllowedTransforms;
		error |= numberOfRigidbodies > settings.maxAllowedRigidbodies;
		error |= numberOfCameras > settings.maxAllowedCameras;
		error |= numberOfMeshRenderers > settings.maxAllowedMeshRenderers;
		error |= numberOfDirectionalLights > settings.maxAllowedDirectionalLights;
		error |= numberOfSpotLights > settings.maxAllowedSpotLights;
		error |= numberOfPointLights > settings.maxAllowedPointLights;

		if(error){
			std::cout << "Error: Number of entities or components exceeds maximum allowed. Please increase max allowed in scene settings." << std::endl;
			return;
		}

		error = settings.maxAllowedEntities <= 0;
		error |= settings.maxAllowedTransforms <= 0;
		error |= settings.maxAllowedRigidbodies <= 0;
		error |= settings.maxAllowedCameras <= 0;
		error |= settings.maxAllowedMeshRenderers <= 0;
		error |= settings.maxAllowedDirectionalLights <= 0;
		error |= settings.maxAllowedSpotLights <= 0;
		error |= settings.maxAllowedPointLights <= 0;

		if(error){
			std::cout << "Error: Total number of entities and components must be strictly greater than zero. Please increase max allowed in scene settings." << std::endl;
			return;
		}

		// de-serialize entities and components
		bytesRead = fread(&entities[existingNumberOfEntities], sceneHeader.numberOfEntities*sizeof(Entity), 1, file);
		bytesRead = fread(&transforms[existingNumberOfTransforms], sceneHeader.numberOfTransforms*sizeof(Transform), 1, file);
		bytesRead = fread(&rigidbodies[existingNumberOfRigidbodies], sceneHeader.numberOfRigidbodies*sizeof(Rigidbody), 1, file);
		bytesRead = fread(&cameras[existingNumberOfCameras], sceneHeader.numberOfCameras*sizeof(Camera), 1, file);
		bytesRead = fread(&meshRenderers[existingNumberOfMeshRenderers], sceneHeader.numberOfMeshRenderers*sizeof(MeshRenderer), 1, file);
		bytesRead = fread(&directionalLights[existingNumberOfDirectionalLights], sceneHeader.numberOfDirectionalLights*sizeof(DirectionalLight), 1, file);
		bytesRead = fread(&spotLights[existingNumberOfSpotLights], sceneHeader.numberOfSpotLights*sizeof(SpotLight), 1, file);
		bytesRead = fread(&pointLights[existingNumberOfPointLights], sceneHeader.numberOfPointLights*sizeof(PointLight), 1, file);

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
	for(int i = 0; i < numberOfDirectionalLights; i++){ directionalLights[i].setManager(this); }
	for(int i = 0; i < numberOfSpotLights; i++){ spotLights[i].setManager(this); }
	for(int i = 0; i < numberOfPointLights; i++){ pointLights[i].setManager(this); }

	// map entity/component id to its global array index
	for(int i = 0; i < numberOfEntities; i++){ idToGlobalIndexMap[entities[i].entityId] = i; }
	for(int i = 0; i < numberOfTransforms; i++){ idToGlobalIndexMap[transforms[i].componentId] = i; }
	for(int i = 0; i < numberOfRigidbodies; i++){ idToGlobalIndexMap[rigidbodies[i].componentId] = i; }
	for(int i = 0; i < numberOfCameras; i++){ idToGlobalIndexMap[cameras[i].componentId] = i; }
	for(int i = 0; i < numberOfMeshRenderers; i++){ idToGlobalIndexMap[meshRenderers[i].componentId] = i; }
	for(int i = 0; i < numberOfDirectionalLights; i++){ idToGlobalIndexMap[directionalLights[i].componentId] = i; }
	for(int i = 0; i < numberOfSpotLights; i++){ idToGlobalIndexMap[spotLights[i].componentId] = i; }
	for(int i = 0; i < numberOfPointLights; i++){ idToGlobalIndexMap[pointLights[i].componentId] = i; }

	// map component id to its type
	for(int i = 0; i < numberOfTransforms; i++){ componentIdToTypeMap[transforms[i].componentId] = (int)ComponentType::TransformType; }
	for(int i = 0; i < numberOfRigidbodies; i++){ componentIdToTypeMap[rigidbodies[i].componentId] = (int)ComponentType::RigidbodyType; }
	for(int i = 0; i < numberOfCameras; i++){ componentIdToTypeMap[cameras[i].componentId] = (int)ComponentType::CameraType; }
	for(int i = 0; i < numberOfMeshRenderers; i++){ componentIdToTypeMap[meshRenderers[i].componentId] = (int)ComponentType::MeshRendererType; }
	for(int i = 0; i < numberOfDirectionalLights; i++){ componentIdToTypeMap[directionalLights[i].componentId] = (int)ComponentType::DirectionalLightType; }
	for(int i = 0; i < numberOfSpotLights; i++){ componentIdToTypeMap[spotLights[i].componentId] = (int)ComponentType::SpotLightType; }
	for(int i = 0; i < numberOfPointLights; i++){ componentIdToTypeMap[pointLights[i].componentId] = (int)ComponentType::PointLightType; }






	// set global indices in entities
	// for(int i = 0; i < numberOfEntities; i++){
	// 	entities[i].globalEntityIndex = i;

	// 	for(int j = 0; j < 8; j++){
	// 		int componentId = entities[i].componentIds[j];
	// 		if(componentId != -1){
	// 			int globalComponentIndex = idToGlobalIndexMap.find(componentId)->second;
	// 			int componentType = componentIdToTypeMap.find(componentId)->second;

	// 			entities[i].globalComponentIndices[j] = globalComponentIndex;
	// 			entities[i].componentTypes[j] = componentType;
	// 		}
	// 	}
	// }

	// set global indices in components
	// setGlobalIndexOnComponent<Transform>(transforms, numberOfTransforms);
	// setGlobalIndexOnComponent<Rigidbody>(rigidbodies, numberOfRigidbodies);
	// setGlobalIndexOnComponent<Camera>(cameras, numberOfCameras);
	// setGlobalIndexOnComponent<MeshRenderer>(meshRenderers, numberOfMeshRenderers);
	// setGlobalIndexOnComponent<DirectionalLight>(directionalLights, numberOfDirectionalLights);
	// setGlobalIndexOnComponent<SpotLight>(spotLights, numberOfSpotLights);
	// setGlobalIndexOnComponent<PointLight>(pointLights, numberOfPointLights);

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

	// find all unique textures and shaders 
	std::vector<int> textureIds;
	std::vector<int> shaderIds;
	for(unsigned int i = 0; i < materialIds.size(); i++){
		bool textureIdFound = false;
		for(unsigned int j = 0; j < textureIds.size(); j++){
			if(materials[i].textureId == textureIds[j]){
				textureIdFound = true;
				break;
			}
		}

		if(!textureIdFound){
			textureIds.push_back(materials[i].textureId);
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

		int size = (width) * (height) * (numChannels);

		std::cout << "size: " << size << " width: " << width << " height: " << height << " num channels: " << numChannels << std::endl;

		std::vector<unsigned char> data;
		data.resize(size);

		textures[i].textureId = textureId;
		//textures[i].globalIndex = i;
		textures[i].setRawTextureData(data);

		stbi_image_free(raw);
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
	    //shaders[i].globalIndex = i;
	    shaders[i].vertexShader = vertexShader;
	    shaders[i].geometryShader = geometryShader;
	    shaders[i].fragmentShader = fragmentShader;
	}

	// set global material, shader, and texture indices 
	// for(unsigned int i = 0; i < materialIds.size(); i++){
	// 	materials[i].globalMaterialIndex = i;
	// 	materials[i].globalShaderIndex = assetIdToGlobalIndexMap[materials[i].shaderId];
	// 	materials[i].globalTextureIndex = assetIdToGlobalIndexMap[materials[i].textureId];
	// }

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
			//meshes[i].globalIndex = i;

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

	// set global mesh and material index on mesh renderers
	// for(int i = 0; i < numberOfMeshRenderers; i++){
	// 	meshRenderers[i].meshGlobalIndex = assetIdToGlobalIndexMap[meshRenderers[i].meshId];
	// 	meshRenderers[i].materialGlobalIndex = assetIdToGlobalIndexMap[meshRenderers[i].materialId];
	// }	
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

Material* Manager::getMaterial(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &materials[it->second];
	}
	else{
		std::cout << "Error: No material with id " << id << " was found" << std::endl;
		return NULL;
	}
}

Shader* Manager::getShader(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &shaders[it->second];
	}
	else{
		std::cout << "Error: No shader with id " << id << " was found" << std::endl;
		return NULL;
	}
}

Texture2D* Manager::getTexture2D(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &textures[it->second];
	}
	else{
		std::cout << "Error: No texture with id " << id << " was found" << std::endl;
		return NULL;
	}
}

Mesh* Manager::getMesh(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &meshes[it->second];
	}
	else{
		std::cout << "Error: No mesh with id " << id << " was found" << std::endl;
		return NULL;
	}
}

GMesh* Manager::getGMesh(int id)
{
	std::map<int, int>::iterator it = idToGlobalIndexMap.find(id);
	if(it != idToGlobalIndexMap.end()){
		return &gmeshes[it->second];
	}
	else{
		std::cout << "Error: No gmesh with id " << id << " was found" << std::endl;
		return NULL;
	}
}
























































// #include "Manager.h"
// #include "../MeshLoader.h"
// #include "../TextureLoader.h"

// using namespace PhysicsEngine;

// Manager::Manager()
// {

// }

// Manager::~Manager()
// {

// }

// Entity* Manager::createEntity()
// {
// 	Entity *entity = entityPool.getNext();

// 	entity->globalEntityIndex = (int)entities.size();
// 	entities.push_back(entity);
	
// 	return entity;
// }

// //void Manager::destroyEntity(unsigned int index)
// //{
// //	if (index == entities.size()){ return; }
// //
// //	entityPool.swapWithLast(index);
// //
// //	// fix this components indicies now that it has been moved from the end to position index
// //	//Entity *entity = entities[index];
// //
// //	entities.pop_back();
// //}

// Transform* Manager::createTransform()
// {
// 	Transform* transform = transformPool.getNext();

// 	transform->globalComponentIndex = (int)transforms.size();
// 	transforms.push_back(transform);

// 	return transform;
// }

// Rigidbody* Manager::createRigidbody()
// {
// 	Rigidbody* rigidbody = rigidbodyPool.getNext();

// 	rigidbody->globalComponentIndex = (int)rigidbodies.size();
// 	rigidbodies.push_back(rigidbody);

// 	return rigidbody;
// }

// DirectionalLight* Manager::createDirectionalLight()
// {
// 	DirectionalLight* light = directionalLightPool.getNext();

// 	light->globalComponentIndex = (int)directionalLights.size();
// 	directionalLights.push_back(light);

// 	return light;
// }

// PointLight* Manager::createPointLight()
// {
// 	PointLight* light = pointLightPool.getNext();

// 	light->globalComponentIndex = (int)pointLights.size();
// 	pointLights.push_back(light);

// 	return light;
// }

// SpotLight* Manager::createSpotLight()
// {
// 	SpotLight* light = spotLightPool.getNext();

// 	light->globalComponentIndex = (int)spotLights.size();
// 	spotLights.push_back(light);

// 	return light;
// }

// MeshRenderer* Manager::createMeshRenderer()
// {
// 	MeshRenderer* mesh = meshRendererPool.getNext();

// 	mesh->globalComponentIndex = (int)meshRenderers.size();
// 	meshRenderers.push_back(mesh);

// 	return mesh;
// }

// // LineRenderer* Manager::createLineRenderer()
// // {
// // 	LineRenderer* line = lineRendererPool.getNext();

// // 	line->globalComponentIndex = (int)lineRenderers.size();
// // 	lineRenderers.push_back(line);

// // 	return line;
// // }

// SphereCollider* Manager::createSphereCollider()
// {
// 	SphereCollider* collider = sphereColliderPool.getNext();

// 	collider->globalComponentIndex = (int)sphereColliders.size();
// 	sphereColliders.push_back(collider);

// 	colliders.push_back(collider);

// 	return collider;
// }

// BoxCollider* Manager::createBoxCollider()
// {
// 	BoxCollider* collider = boxColliderPool.getNext();

// 	collider->globalComponentIndex = (int)boxColliders.size();
// 	boxColliders.push_back(collider);

// 	colliders.push_back(collider);

// 	return collider;
// }

// SpringJoint* Manager::createSpringJoint()
// {
// 	SpringJoint* joint = springJointPool.getNext();

// 	joint->globalComponentIndex = (int)springJoints.size();
// 	springJoints.push_back(joint);

// 	joints.push_back(joint);

// 	return joint;
// }

// Fluid* Manager::createFluid()
// {
// 	Fluid* fluid = fluidPool.getNext();

// 	fluid->globalComponentIndex = (int)fluids.size();
// 	fluids.push_back(fluid);

// 	return fluid;
// }

// Cloth* Manager::createCloth()
// {
// 	Cloth* cloth = clothPool.getNext();

// 	cloth->globalComponentIndex = (int)cloths.size();
// 	cloths.push_back(cloth);

// 	return cloth;
// }

// Solid* Manager::createSolid()
// {
// 	Solid* solid = solidPool.getNext();

// 	solid->globalComponentIndex = (int)solids.size();
// 	solids.push_back(solid);

// 	return solid;
// }

// Camera* Manager::createCamera()
// {
// 	Camera* camera = cameraPool.getNext();

// 	this->camera = camera;

// 	return camera;
// }

// std::vector<Entity*> Manager::getEntities()
// {
// 	return entities;
// 	//return entityPool.getPool();
// }

// std::vector<Transform*> Manager::getTransforms()
// {
// 	return transforms;
// 	//return transformPool.getPool();
// }

// std::vector<Rigidbody*> Manager::getRigidbodies()
// {
// 	return rigidbodies;
// 	//return rigidbodyPool.getPool();
// }

// std::vector<DirectionalLight*> Manager::getDirectionalLights()
// {
// 	return directionalLights;
// 	//return directionalLightPool.getPool();
// }

// std::vector<PointLight*> Manager::getPointLights()
// {
// 	return pointLights;
// 	//return pointLightPool.getPool();
// }

// std::vector<SpotLight*> Manager::getSpotLights()
// {
// 	return spotLights;
// 	//return spotLightPool.getPool();
// }

// std::vector<MeshRenderer*> Manager::getMeshRenderers()
// {
// 	return meshRenderers;
// 	//return meshRendererPool.getPool();
// }

// // std::vector<LineRenderer*> Manager::getLineRenderers()
// // {
// // 	return lineRenderers;
// // 	//return lineRendererPool.getPool();
// // }

// std::vector<Collider*> Manager::getColliders()
// {
// 	return colliders;
// }

// std::vector<SphereCollider*> Manager::getSphereColliders()
// {
// 	return sphereColliders;
// }

// std::vector<BoxCollider*> Manager::getBoxColliders()
// {
// 	return boxColliders;
// }

// std::vector<Joint*> Manager::getJoints()
// {
// 	return joints;
// }

// std::vector<SpringJoint*> Manager::getSpringJoints()
// {
// 	return springJoints;
// }

// std::vector<Fluid*> Manager::getFluids()
// {
// 	return fluids;
// 	//return fluidPool.getPool();
// }

// std::vector<Cloth*> Manager::getCloths()
// {
// 	return cloths;
// 	//return clothPool.getPool();
// }

// std::vector<Solid*> Manager::getSolids()
// {
// 	return solids;
// 	//return clothPool.getPool();
// }

// Camera* Manager::getCamera()
// {
// 	return camera;
// 	//return editorCameraPool.getPool();
// }

// void Manager::loadGMesh(const std::string& name)
// {
// 	if (gmeshMap.count(name) != 0){
// 		std::cout << "gmesh: " << name << " already loaded" << std::endl;
// 		return;
// 	}

// 	std::cout << "loading gmesh: " << name << std::endl;

// 	GMesh gmesh;
// 	if (MeshLoader::load_gmesh(name, gmesh)){
// 		gmeshes.push_back(gmesh);

// 		gmeshMap[name] = (int)gmeshes.size() - 1;
// 	}
// 	else{
// 		std::cout << "Could not load gmesh " << name << std::endl;
// 	}
// }

// void Manager::loadMesh(const std::string& name)
// {
// 	if (meshMap.count(name) != 0){
// 		std::cout << "mesh: " << name << " already loaded" << std::endl;
// 		return;
// 	}

// 	std::cout << "loading mesh: " << name << std::endl;

// 	Mesh mesh;
// 	if (MeshLoader::load(name, mesh)){
// 		meshes.push_back(mesh);

// 		meshMap[name] = (int)meshes.size() - 1;
// 	}
// 	else{
// 		std::cout << "Could not load mesh " << name << std::endl;
// 	}
// }

// void Manager::loadTexture2D(const std::string& name)
// {
// 	if (textureMap.count(name) != 0){
// 		std::cout << "texture: " << name << " already loaded" << std::endl;
// 		return;
// 	}
// 	int width, height, numChannels;
// 	std::vector<unsigned char> rawTextureData;
	
// 	if (TextureLoader::load(name, rawTextureData, &width, &height, &numChannels)){
		
// 		TextureFormat format = Red;
// 		if (numChannels == 3){
// 			format = TextureFormat::RGB;
// 		}
// 		else if (numChannels == 4){
// 			format = TextureFormat::RGBA;
// 		}
// 		else{
// 			std::cout << "Manager: Number of channels not supported" << std::endl;
// 			return;
// 		}

// 		Texture2D texture(width, height, format);
// 		texture.setRawTextureData(rawTextureData);

// 		textures.push_back(texture);

// 		textureMap[name] = (int)textures.size() - 1;
// 	}
// 	else{
// 		std::cout << "Could not load texture " << name << std::endl;
// 	}
// }

// void Manager::loadCubemap(const std::vector<std::string>& names)
// {
// 	std::string name;
// 	for (unsigned int i = 0; i < names.size(); i++){
// 		name += names[i];
// 	}

// 	if (cubemapMap.count(name) != 0){
// 		std::cout << "Manager: Cubemap texture: " << name << " already loaded" << std::endl;
// 		return;
// 	}

// 	if (names.size() != 6){
// 		std::cout << "Manager: When loading cubemaps, exactly 6 filenames must be passed" << std::endl;
// 		return;
// 	}

// 	int width, height, numChannels;
// 	std::vector<unsigned char> rawCubemapData;

// 	for (unsigned int i = 0; i < 6; i++){
// 		std::vector<unsigned char> data;

// 		if (!TextureLoader::load(names[i], data, &width, &height, &numChannels)){
// 			std::cout << "Manager: Could not load " << i << "th image of cubemap " << names[i] << std::endl;
// 			return;
// 		}

// 		for (unsigned int j = 0; j < data.size(); j++){
// 			rawCubemapData.push_back(data[j]);
// 		}
// 	}

// 	if (rawCubemapData.size() != 6*width*height*numChannels){
// 		std::cout << "Manager: each face of the cubemap must have the same size and channels" << std::endl;
// 		return;
// 	}

// 	TextureFormat format = Red;
// 	if (numChannels == 3){
// 		format = TextureFormat::RGB;
// 	}
// 	else if (numChannels == 4){
// 		format = TextureFormat::RGBA;
// 	}
// 	else{
// 		std::cout << "Manager: Number of channels not supported" << std::endl;
// 		return;
// 	}

// 	Cubemap cubemap(width, format);

// 	cubemap.setRawCubemapData(rawCubemapData);

// 	cubemaps.push_back(cubemap);

// 	cubemapMap[name] = (int)cubemaps.size() - 1;
// }

// void Manager::loadShader(const std::string& name, std::string vertex, std::string fragment, std::string geometry)
// {
// 	if (shaderMap.count(name) != 0){
// 		std::cout << "shader program: " << name << " already loaded" << std::endl;
// 		return;
// 	}

// 	std::cout << "loading shader program: " << name << std::endl;

// 	shaders.push_back(Shader(vertex, fragment, geometry));

// 	shaderMap[name] = (int)shaders.size() - 1;
// }

// void Manager::loadMaterial(const std::string& name, Material mat)
// {
// 	if (materialMap.count(name) != 0){
// 		std::cout << "material " << name << " already loaded" << std::endl;
// 		return;
// 	}

// 	std::cout << "loading material: " << name << std::endl;

// 	materials.push_back(mat);

// 	materialMap[name] = (int)materials.size() - 1;
// }

// GMesh* Manager::getGMesh(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = gmeshMap.find(name);
// 	if (it != gmeshMap.end()){
// 		return &gmeshes[it->second];
// 	}

// 	return NULL;
// }

// Mesh* Manager::getMesh(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = meshMap.find(name);
// 	if (it != meshMap.end()){
// 		return &meshes[it->second];
// 	}

// 	return NULL;
// }

// Texture2D* Manager::getTexture2D(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = textureMap.find(name);
// 	if (it != textureMap.end()){
// 		return &textures[it->second];
// 	}

// 	return NULL;
// }

// Cubemap* Manager::getCubemap(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = cubemapMap.find(name);
// 	if (it != cubemapMap.end()){
// 		return &cubemaps[it->second];
// 	}

// 	return NULL;
// }

// Shader* Manager::getShader(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = shaderMap.find(name);
// 	if (it != shaderMap.end()){
// 		return &shaders[it->second];
// 	}

// 	return NULL;
// }

// Material* Manager::getMaterial(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = materialMap.find(name);
// 	if (it != materialMap.end()){
// 		return &materials[it->second];
// 	}

// 	return NULL;
// }

// std::vector<GMesh>& Manager::getGMeshes()
// {
// 	return gmeshes;
// }

// std::vector<Mesh>& Manager::getMeshes()
// {
// 	return meshes;
// }

// std::vector<Texture2D>& Manager::getTextures()
// {
// 	return textures;
// }

// std::vector<Cubemap>& Manager::getCubemaps()
// {
// 	return cubemaps;
// }

// std::vector<Shader>& Manager::getShaders()
// {
// 	return shaders;
// }

// std::vector<Material>& Manager::getMaterials()
// {
// 	return materials;
// }

// int Manager::getGMeshFilter(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = gmeshMap.find(name);
// 	if (it != gmeshMap.end()){
// 		return it->second;
// 	}

// 	return -1;
// }

// int Manager::getMeshFilter(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = meshMap.find(name);
// 	if (it != meshMap.end()){
// 		return it->second;
// 	}

// 	return -1;
// }

// int Manager::getMaterialFilter(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = materialMap.find(name);
// 	if (it != materialMap.end()){
// 		return it->second;
// 	}

// 	return -1;
// }

// GMesh* Manager::getGMesh(int filter)
// {
// 	if (filter < 0 || filter >= (int)gmeshes.size()){
// 		std::cout << "Invalid gmesh filter: " << filter << std::endl;
// 	}

// 	return &gmeshes[filter];
// }

// Mesh* Manager::getMesh(int filter)
// {
// 	if (filter < 0 || filter >= (int)meshes.size()){
// 		std::cout << "Invalid mesh filter: " << filter << std::endl;
// 	}

// 	return &meshes[filter];
// }

// Material* Manager::getMaterial(int filter)
// {
// 	if (filter < 0 || filter >= (int)materials.size()){
// 		std::cout << "Invalid material filter: " << filter << std::endl;
// 	}

// 	return &materials[filter];
// }