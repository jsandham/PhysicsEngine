#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

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

int Manager::load(std::string &sceneFilePath, std::vector<std::string> &assetFilePaths)
{
	// create asset id to filepath map
	for(unsigned int i = 0; i < assetFilePaths.size(); i++){
		// open asset files json object and get asset id
		std::string jsonAssetFilePath = assetFilePaths[i].substr(0, assetFilePaths[i].find_last_of(".")) + ".json";
		std::cout << "jsonAssetFilePath: " << jsonAssetFilePath << std::endl;
		std::ifstream in(jsonAssetFilePath, std::ios::in | std::ios::binary);
		std::ostringstream contents;
		contents << in.rdbuf();
		in.close();

		std::string jsonString = contents.str();
		json::JSON jsonAsset = JSON::Load(jsonString);

		int assetId = jsonAsset["id"].ToInt();

		std::cout << "asset id: " << assetId << " file path: " << assetFilePaths[i] << std::endl;
		
		assetIdToFilePathMap[assetId] = assetFilePaths[i];
	}

	std::string binarySceneFilePath = sceneFilePath.substr(0, sceneFilePath.find_last_of(".")) + ".scene";

	std::cout << "scene file path: " << sceneFilePath << " binary scene file path: " << binarySceneFilePath << " size of camera: " << sizeof(Camera) << std::endl;

	SceneHeader sceneHeader = {};
	FILE* file = fopen(binarySceneFilePath.c_str(), "rb");
	size_t bytesRead;
	if (file){
		bytesRead = fread(&sceneHeader, sizeof(SceneHeader), 1, file);

		std::cout << "de-serialized scene header file contains the following information: " << std::endl;

		std::cout << "numberOfEntities: " << sceneHeader.numberOfEntities << std::endl;
		std::cout << "numberOfTransforms: " << sceneHeader.numberOfTransforms << std::endl;
		std::cout << "numberOfRigidbodies: " << sceneHeader.numberOfRigidbodies << std::endl;
		std::cout << "numberOfCameras: " << sceneHeader.numberOfCameras << std::endl;
		std::cout << "numberOfMeshRenderers: " << sceneHeader.numberOfMeshRenderers << std::endl;
		std::cout << "numberOfDirectionalLights: " << sceneHeader.numberOfDirectionalLights << std::endl;
		std::cout << "numberOfSpotLights: " << sceneHeader.numberOfSpotLights << std::endl;
		std::cout << "numberOfPointLights: " << sceneHeader.numberOfPointLights << std::endl;

		std::cout << "sizeOfEntity: " << sceneHeader.sizeOfEntity << std::endl;
		std::cout << "sizeOfTransform: " << sceneHeader.sizeOfTransform << std::endl;
		std::cout << "sizeOfRigidbodies: " << sceneHeader.sizeOfRigidbody << std::endl;
		std::cout << "sizeOfCameras: " << sceneHeader.sizeOfCamera << std::endl;
		std::cout << "sizeOfMeshRenderer: " << sceneHeader.sizeOfMeshRenderer << std::endl;
		std::cout << "sizeOfDirectionalLight: " << sceneHeader.sizeOfDirectionalLight << std::endl;
		std::cout << "sizeOfSpotLight: " << sceneHeader.sizeOfSpotLight << std::endl;
		std::cout << "sizeOfPointLight: " << sceneHeader.sizeOfPointLight << std::endl;

		numberOfEntities = sceneHeader.numberOfEntities;
		numberOfTransforms = sceneHeader.numberOfTransforms;
		numberOfRigidbodies = sceneHeader.numberOfRigidbodies;
		numberOfCameras = sceneHeader.numberOfCameras;
		numberOfMeshRenderers = sceneHeader.numberOfMeshRenderers;
		numberOfDirectionalLights = sceneHeader.numberOfDirectionalLights;
		numberOfSpotLights = sceneHeader.numberOfSpotLights;
		numberOfPointLights = sceneHeader.numberOfPointLights;

		bytesRead = fread(&settings, sizeof(SceneSettings), 1, file);

		totalNumberOfEntitiesAlloc = settings.maxAllowedEntities;
		totalNumberOfTransformsAlloc = settings.maxAllowedTransforms;
		totalNumberOfRigidbodiesAlloc = settings.maxAllowedRigidbodies;
		totalNumberOfCamerasAlloc = settings.maxAllowedCameras;
		totalNumberOfMeshRenderersAlloc = settings.maxAllowedMeshRenderers;
		totalNumberOfDirectionalLightsAlloc= settings.maxAllowedDirectionalLights;
		totalNumberOfSpotLightsAlloc = settings.maxAllowedSpotLights;
		totalNumberOfPointLightsAlloc = settings.maxAllowedPointLights;

		std::cout << "Total number of entities alloc: " << totalNumberOfEntitiesAlloc << std::endl;
		std::cout << "Total number of transforms alloc: " << totalNumberOfTransformsAlloc << std::endl;
		std::cout << "Total number of rigidbodies alloc: " << totalNumberOfRigidbodiesAlloc << std::endl;
		std::cout << "Total number of cameras alloc: " << totalNumberOfCamerasAlloc << std::endl;
		std::cout << "Total number of mesh renderers alloc: " << totalNumberOfMeshRenderersAlloc << std::endl;
		std::cout << "Total number of directional lights alloc: " << totalNumberOfDirectionalLightsAlloc << std::endl;
		std::cout << "Total number of spot lights alloc: " << totalNumberOfSpotLightsAlloc << std::endl;
		std::cout << "Total number of point lights alloc: " << totalNumberOfPointLightsAlloc << std::endl;

		bool error = numberOfEntities > totalNumberOfEntitiesAlloc;
		error |= numberOfTransforms > totalNumberOfTransformsAlloc;
		error |= numberOfRigidbodies > totalNumberOfRigidbodiesAlloc;
		error |= numberOfCameras > totalNumberOfCamerasAlloc;
		error |= numberOfMeshRenderers > totalNumberOfMeshRenderersAlloc;
		error |= numberOfDirectionalLights > totalNumberOfDirectionalLightsAlloc;
		error |= numberOfSpotLights > totalNumberOfSpotLightsAlloc;
		error |= numberOfPointLights > totalNumberOfPointLightsAlloc;

		if(error){
			std::cout << "Error: Number of entities or components exceeds maximum allowed. Please increase max allowed in scene settings." << std::endl;
			return 0;
		}

		error = totalNumberOfEntitiesAlloc <= 0;
		error |= totalNumberOfTransformsAlloc <= 0;
		error |= totalNumberOfRigidbodiesAlloc <= 0;
		error |= totalNumberOfCamerasAlloc <= 0;
		error |= totalNumberOfMeshRenderersAlloc <= 0;
		error |= totalNumberOfDirectionalLightsAlloc <= 0;
		error |= totalNumberOfPointLightsAlloc <= 0;

		if(error){
			std::cout << "Error: Total number of entities and components must be strictly greater than zero. Please increase max allowed in scene settings." << std::endl;
			return 0;
		}

		// allocate memory blocks for entities and components
		entities = new Entity[totalNumberOfEntitiesAlloc];
		transforms = new Transform[totalNumberOfTransformsAlloc];
		rigidbodies = new Rigidbody[totalNumberOfRigidbodiesAlloc];
		cameras = new Camera[totalNumberOfCamerasAlloc];
		meshRenderers = new MeshRenderer[totalNumberOfMeshRenderersAlloc];
		directionalLights = new DirectionalLight[totalNumberOfDirectionalLightsAlloc];
		spotLights = new SpotLight[totalNumberOfSpotLightsAlloc];
		pointLights = new PointLight[totalNumberOfPointLightsAlloc];

		// de-serialize entities and components
		bytesRead = fread(&entities[0], numberOfEntities*sizeof(Entity), 1, file);
		bytesRead = fread(&transforms[0], numberOfTransforms*sizeof(Transform), 1, file);
		bytesRead = fread(&rigidbodies[0], numberOfRigidbodies*sizeof(Rigidbody), 1, file);
		bytesRead = fread(&cameras[0], numberOfCameras*sizeof(Camera), 1, file);
		bytesRead = fread(&meshRenderers[0], numberOfMeshRenderers*sizeof(MeshRenderer), 1, file);
		bytesRead = fread(&directionalLights[0], numberOfDirectionalLights*sizeof(DirectionalLight), 1, file);
		bytesRead = fread(&spotLights[0], numberOfSpotLights*sizeof(SpotLight), 1, file);
		bytesRead = fread(&pointLights[0], numberOfPointLights*sizeof(PointLight), 1, file);

		for(int i = 0; i < numberOfEntities; i++){
			for(int j = 0; j < 8; j++){
				std::cout << "component types: " << entities[i].componentTypes[j] << " globalComponentIndices: " << entities[i].globalComponentIndices[j] << std::endl;
			}
		}

		std::cout << "number of bytes read from file: " << bytesRead << std::endl;

		fclose(file);
	}
	else{
		std::cout << "Error: Failed to open scene binary file " << binarySceneFilePath << " for reading" << std::endl;
		return 0;
	}

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
	for(int i = 0; i < numberOfEntities; i++){
		entities[i].globalEntityIndex = i;

		for(int j = 0; j < 8; j++){
			int componentId = entities[i].componentIds[j];
			if(componentId != -1){
				int globalComponentIndex = idToGlobalIndexMap.find(componentId)->second;
				int componentType = componentIdToTypeMap.find(componentId)->second;

				entities[i].globalComponentIndices[j] = globalComponentIndex;
				entities[i].componentTypes[j] = componentType;
			}
		}
	}

	// set global indices in components
	setGlobalIndexOnComponent<Transform>(transforms, numberOfTransforms);
	setGlobalIndexOnComponent<Rigidbody>(rigidbodies, numberOfRigidbodies);
	setGlobalIndexOnComponent<Camera>(cameras, numberOfCameras);
	setGlobalIndexOnComponent<MeshRenderer>(meshRenderers, numberOfMeshRenderers);
	setGlobalIndexOnComponent<DirectionalLight>(directionalLights, numberOfDirectionalLights);
	setGlobalIndexOnComponent<SpotLight>(spotLights, numberOfSpotLights);
	setGlobalIndexOnComponent<PointLight>(pointLights, numberOfPointLights);

	for(int i = 0; i < numberOfEntities; i++){
		for(int j = 0; j < 8; j++){
			std::cout << "component types: " << entities[i].componentTypes[j] << " globalComponentIndices: " << entities[i].globalComponentIndices[j] << std::endl;
		}
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
			materialIds.push_back(meshRenderers[i].materialId);
		}
	}

	totalNumberOfMaterialsAlloc = (int)materialIds.size();

	// allocate materials and meshes
	materials = new Material[totalNumberOfMaterialsAlloc];

	// de-serialize all unique materials found
	for(unsigned int i = 0; i < materialIds.size(); i++){
		int materialId = materialIds[i];

		assetIdToGlobalIndexMap[materialId] = i;

		std::string materialFilePath = assetIdToFilePathMap[materialId];

		FILE* file = fopen(materialFilePath.c_str(), "rb");
		size_t bytesRead;
		if (file){
			bytesRead = fread(&materials[i], sizeof(Material), 1, file);
			std::cout << "number of bytes read from file: " << bytesRead << std::endl;
		}
		else{
			std::cout << "Error: Failed to open material binary file " << materialFilePath << " for reading" << std::endl;
			return 0;
		}

		std::cout << "material id: " << materials[i].materialId << " texture id: " << materials[i].textureId << " shader id: " << materials[i].shaderId << std::endl;
	}

	std::cout << "numberOfMeshRenderers: " << numberOfMeshRenderers << std::endl;

	// find all unique textures and shaders 
	std::vector<int> textureIds;
	std::vector<int> shaderIds;
	for(int i = 0; i < totalNumberOfMaterialsAlloc; i++){
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

	totalNumberOfTexturesAlloc = (int)textureIds.size();
	totalNumberOfShadersAlloc = (int)shaderIds.size();

	std::cout << "total number of textures alloc: " << totalNumberOfTexturesAlloc << " total number of shaders alloc: " << totalNumberOfShadersAlloc << std::endl;

	// allocate textures and shaders
	textures = new Texture2D[totalNumberOfTexturesAlloc];
	shaders = new Shader[totalNumberOfShadersAlloc];

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

		for (int j = 0; j < size; j++){
			data[j] = raw[j];
			std::cout << (int)data[j] << " ";
		}
		std::cout << "" << std::endl;

		textures[i].textureId = textureId;
		textures[i].globalIndex = i;
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
	    std::ostringstream contents;

	    contents << in.rdbuf();
	    std::string shader = contents.str();
	    in.close();

	    std::string vertexTag = "VERTEX:";
	    std::string geometryTag = "GEOMETRY:";
	    std::string fragmentTag = "FRAGMENT:";

	    size_t startOfVertexTag = shader.find(vertexTag, 0);
	    size_t startOfGeometryTag = shader.find(geometryTag, 0);
	    size_t startOfFragmentTag = shader.find(fragmentTag, 0);

	    if(startOfVertexTag == std::string::npos || startOfFragmentTag == std::string::npos){
	    	std::cout << "Error: Shader must contain both a vertex shader and a fragment shader" << std::endl;
	    	return 0;
	    }

	    std::string vertexShader, geometryShader, fragmentShader;
	
	    if(startOfGeometryTag == std::string::npos){
	    	std::cout << "BBBBBBBBBBB" << std::endl;
	    	vertexShader = shader.substr(startOfVertexTag + vertexTag.length(), startOfFragmentTag - vertexTag.length());
	    	geometryShader = "";
	    	fragmentShader = shader.substr(startOfFragmentTag + fragmentTag.length(), shader.length());
	    }
	    else{
	    	vertexShader = shader.substr(startOfVertexTag + vertexTag.length(), startOfGeometryTag - vertexTag.length());
	    	geometryShader = shader.substr(startOfGeometryTag + geometryTag.length(), startOfFragmentTag - geometryTag.length());
	    	fragmentShader = shader.substr(startOfFragmentTag + fragmentTag.length(), shader.length());
	    }

	    // std::cout << "shader: " << shader << std::endl;
	    std::cout << "start of vertex shader: " << startOfVertexTag  << " start of fragment shader: " << startOfFragmentTag << " start of geometry shader: " << startOfGeometryTag << std::endl;
	    std::cout << "vertexShader: " << vertexShader << std::endl;
	    //std::cout << "fragmentShader: " << fragmentShader << std::endl;

	    shaders[i].shaderId = shaderId;
	    shaders[i].globalIndex = i;
	    shaders[i].vertexShader = vertexShader;
	    shaders[i].geometryShader = geometryShader;
	    shaders[i].fragmentShader = fragmentShader;
	}

	// set global material, shader, and texture indices 
	for(int i = 0; i < totalNumberOfMaterialsAlloc; i++){
		materials[i].globalMaterialIndex = i;
		materials[i].globalShaderIndex = assetIdToGlobalIndexMap[materials[i].shaderId];
		materials[i].globalTextureIndex = assetIdToGlobalIndexMap[materials[i].textureId];
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

	totalNumberOfMeshesAlloc = (int)meshIds.size();

	// allocate meshes
	meshes = new Mesh[totalNumberOfMeshesAlloc];

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
			meshes[i].globalIndex = i;

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
			return 0;
		}

		std::cout << "mesh id: " << meshId << " mesh header number of vertices: " << header.verticesSize << " number of normals: " << header.normalsSize << " number of texCoords: " << header.texCoordsSize << std::endl;
	}

	std::cout << "numberOfMeshRenderers: " << numberOfMeshRenderers << std::endl;

	// set global mesh and material index on mesh renderers
	for(int i = 0; i < numberOfMeshRenderers; i++){
		meshRenderers[i].meshGlobalIndex = assetIdToGlobalIndexMap[meshRenderers[i].meshId];
		meshRenderers[i].materialGlobalIndex = assetIdToGlobalIndexMap[meshRenderers[i].materialId];
	}	

	return 1;
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

Entity* Manager::getEntity(int globalIndex)
{
	if(globalIndex >= numberOfEntities){
		std::cout << "Error: Trying to access entity outside of range" << std::endl;
	}

	return &entities[globalIndex];
}

Transform* Manager::getTransform(int globalIndex)
{
	if(globalIndex >= numberOfTransforms){
		std::cout << "Error: Trying to access transform outside of range" << std::endl;
	}

	return &transforms[globalIndex];
}

Rigidbody* Manager::getRigidbody(int globalIndex)
{
	if(globalIndex >= numberOfRigidbodies){
		std::cout << "Error: Trying to access rigidbody outside of range" << std::endl;
	}

	return &rigidbodies[globalIndex];
}

Camera* Manager::getCamera(int globalIndex)
{
	if(globalIndex >= numberOfCameras){
		std::cout << "Error: Trying to access camera outside of range" << std::endl;
	}

	return &cameras[globalIndex];
}

MeshRenderer* Manager::getMeshRenderer(int globalIndex)
{
	if(globalIndex >= numberOfMeshRenderers){
		std::cout << "Error: Trying to access mesh renderer outside of range" << std::endl;
	}

	return &meshRenderers[globalIndex];
}

DirectionalLight* Manager::getDirectionalLight(int globalIndex)
{
	if(globalIndex >= numberOfDirectionalLights){
		std::cout << "Error: Trying to access directional light outside of range" << std::endl;
	}

	return &directionalLights[globalIndex];
}

SpotLight* Manager::getSpotLight(int globalIndex)
{
	if(globalIndex >= numberOfSpotLights){
		std::cout << "Error: Trying to access spot light outside of range" << std::endl;
	}

	return &spotLights[globalIndex];
}	

PointLight* Manager::getPointLight(int globalIndex)
{
	if(globalIndex >= numberOfPointLights){
		std::cout << "Error: Trying to access point light outside of range" << std::endl;
	}

	return &pointLights[globalIndex];
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