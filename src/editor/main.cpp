#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtx/quaternion.hpp"
#include "../glm/gtc/matrix_transform.hpp"

#include "../entities/Entity.h"
#include "../components/Transform.h"
#include "../components/MeshRenderer.h"
#include "../components/DirectionalLight.h"

#include "../core/Manager.h"

#include "../json/json.hpp"

using namespace json;
using namespace PhysicsEngine;


int serializeScene(std::string scenePath);
int serializeMaterials(std::string materialPath);
int serializeMeshes(std::string meshPath);
int serializeGMeshes(std::string gmeshPath);

int main(int argc, char* argv[])
{
	if(argc > 0){
		std::cout << argv[0] << std::endl;
	}

	std::string scenePath = "../data/scenes/simple.json";
	std::string materialsPath = "../data/materials/materials.json";
	std::string meshesPath = "../data/meshes/meshes.json";
	std::string gmeshesPath = "../data/gmeshes/gmeshes.json";

	if(!serializeScene(scenePath)){
		std::cout << "Failed to serialize scene" << std::endl;
	}

	if(!serializeMaterials(materialsPath)){
		std::cout << "Failed to serialize materials" << std::endl;
	}

	if(!serializeMeshes(meshesPath)){
		std::cout << "Failed to serialize meshes" << std::endl;
	}

	if(!serializeGMeshes(gmeshesPath)){
		std::cout << "Failed to serialize gmeshes" << std::endl;
	}

	while(true)
	{

	}

	return 0;
}


int serializeScene(std::string scenePath)
{
	std::string outputPath = scenePath.substr(0, scenePath.find_last_of(".")) + ".scene";

	// open json file and load to json object
	std::ifstream in(scenePath, std::ios::in | std::ios::binary);
	std::ostringstream contents;
	contents << in.rdbuf();
	in.close();
	std::string jsonString = contents.str();
	json::JSON scene = JSON::Load(jsonString);

	// parse loaded json file
	json::JSON entities;
	json::JSON transforms;
	json::JSON rigidbodies;
	json::JSON meshRenderers;
	json::JSON directionalLights;
	json::JSON spotLights;
	json::JSON pointsLights;

	json::JSON::JSONWrapper<map<string,JSON>> objects = scene.ObjectRange();
	map<string,JSON>::iterator it;

	for(it = objects.begin(); it != objects.end(); it++){
		if(it->first == "Settings"){
			continue;
		}

		std::string type = it->second["type"].ToString();

		if(type == "Entity"){
			std::cout << it->first << " is an Entity" << std::endl;
			entities[it->first] = it->second;
		}
		else if(type == "Transform"){
			std::cout << it->first << " is a Transform" << std::endl;
			transforms[it->first] = it->second;
		}
		else if(type == "Rigidbody"){
			std::cout << it->first << " is a Rigidbody" << std::endl;
			rigidbodies[it->first] = it->second;
		}
		else if(type == "MeshRenderer"){
			meshRenderers[it->first] = it->second;
		}
		else if(type == "DirectionalLight"){
			directionalLights[it->first] = it->second;
		}
		else if(type == "SpotLight"){
			spotLights[it->first] = it->second;
		}
		else if(type == "PointLight"){
			pointLights[it->first] = it->second;
		}
	}

	int numberOfEntities = std::max(0, entities.size());
	int numberOfTransforms = std::max(0, transforms.size());
	int numberOfRigidbodies = std::max(0, rigidbodies.size());
	int numberOfMeshRenderers = std::max(0, meshRenderers.size());
	int numberOfDirectionalLights = std::max(0, directionalLights.size());
	int numberOfSpotLights = std::max(0, spotLights.size());
	int numberOfPointLights = std::max(0, pointLights.size());

	std::cout << "number of entities found: " << numberOfEntities << std::endl;
	std::cout << "number of transforms found: " << numberOfTransforms << std::endl;
	std::cout << "number of rigidbodies found: " << numberOfRigidbodies << std::endl;
	std::cout << "number of mesh renderers found: " << numberOfMeshRenderers << std::endl;
	std::cout << "number of directional lights found: " << numberOfDirectionalLights << std::endl;
	std::cout << "number of spot lights found: " << numberOfSpotLights << std::endl;
	std::cout << "number of point lights found: " << numberOfPointLights << std::endl;

	// create scene header
	SceneHeader header = {};

	header.sizeOfEntity = sizeof(Entity);
	header.sizeOfTransform = sizeof(Transform);
	header.sizeOfMeshRenderer = sizeof(MeshRenderer);
	header.sizeOfDirectionalLight = sizeof(DirectionalLight);
	header.numberOfEntities = numberOfEntities;
	header.numberOfTransforms = numberOfTransforms;
	header.numberOfMeshRenderers = numberOfMeshRenderers;
	header.numberOfDirectionalLights = numberOfDirectionalLights;

	// serialize scene header
	FILE* file = fopen(outputPath.c_str(), "wb");
	if (file){
		size_t test = fwrite(&header, sizeof(SceneHeader), 1, file);
		std::cout << "number of bytes written to file: " << test << std::endl;
	}
	else{
		std::cout << "Failed to open file " << outputPath << " for writing" << std::endl;
		return 0;
	}

	// serialize entities
	objects = entities.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		Entity entity;

		entity.entityId = std::stoi(it->first);

		for(int i = 0; i < it->second["components"].size(); i++){
			entity.componentIds[i] = it->second["components"][i].ToInt();
		}
		fwrite(&entity, sizeof(Entity), 1, file);
	}

	// serialize transforms
	objects = transforms.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		Transform transform;

		transform.componentId = std::stoi(it->first);
		transform.entityId = it->second["entity"].ToInt();

		transform.position.x = (float)it->second["position"][0].ToFloat();
		transform.position.y = (float)it->second["position"][1].ToFloat();
		transform.position.z = (float)it->second["position"][2].ToFloat();

		transform.rotation.x = (float)it->second["rotation"][0].ToFloat();
		transform.rotation.y = (float)it->second["rotation"][1].ToFloat();
		transform.rotation.z = (float)it->second["rotation"][2].ToFloat();
		transform.rotation.z = (float)it->second["rotation"][3].ToFloat();

		transform.scale.x = (float)it->second["scale"][0].ToFloat();
		transform.scale.y = (float)it->second["scale"][1].ToFloat();
		transform.scale.z = (float)it->second["scale"][2].ToFloat();

		fwrite(&transform, sizeof(Transform), 1, file);
	}

	// serialize rigidbodies
	objects = rigidbodies.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		Rigidbody rigidbody;

		rigidbody.componentId = std::stoi(it->first);
		rigidbody.entityId = it->second["entity"].ToInt();

		rigidbody.useGravity = (bool)it->second["useGravity"].ToBool();
		rigidbody.mass = (float)it->second["mass"].ToFloat();
		rigidbody.drag = (float)it->second["drag"].ToFloat();
		rigidbody.angularDrag = (float)it->second["angularDrag"].ToFloat();

		rigidbody.velocity.x = (float)it->second["velocity"][0].ToFloat();
		rigidbody.velocity.y = (float)it->second["velocity"][1].ToFloat();
		rigidbody.velocity.z = (float)it->second["velocity"][2].ToFloat();

		rigidbody.angularVelocity.x = (float)it->second["angularVelocity"][0].ToFloat();
		rigidbody.angularVelocity.y = (float)it->second["angularVelocity"][1].ToFloat();
		rigidbody.angularVelocity.z = (float)it->second["angularVelocity"][2].ToFloat();

		rigidbody.centreOfMass.x = = (float)it->second["centreOfMass"][0].ToFloat();
		rigidbody.centreOfMass.y = = (float)it->second["centreOfMass"][1].ToFloat();
		rigidbody.centreOfMass.z = = (float)it->second["centreOfMass"][2].ToFloat();
		
		//glm::mat3 inertiaTensor;

		fwrite(&rigidbody, sizeof(Rigidbody), 1, file);
	}

	// serialize mesh renderers
	objects = meshRenderers.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		MeshRenderer meshRenderer;

		meshRenderer.componentId = std::stoi(it->first);
		meshRenderer.entityId = it->second["entity"].ToInt();

		meshRenderer.meshFilter = 0;
		meshRenderer.materialFilter = 2;

		fwrite(&meshRenderer, sizeof(MeshRenderer), 1, file);
	}

	// serialize directional lights
	objects = directionalLights.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		DirectionalLight directionalLight;

		directionalLight.componentId = std::stoi(it->first);
		directionalLight.entityId = it->second["entity"].ToInt();

		directionalLight.direction.x = (float)it->second["direction"][0].ToFloat();
		directionalLight.direction.y = (float)it->second["direction"][1].ToFloat();
		directionalLight.direction.z = (float)it->second["direction"][2].ToFloat();

		directionalLight.ambient.x = (float)it->second["ambient"][0].ToFloat();
		directionalLight.ambient.y = (float)it->second["ambient"][1].ToFloat();
		directionalLight.ambient.z = (float)it->second["ambient"][2].ToFloat();

		directionalLight.diffuse.x = (float)it->second["diffuse"][0].ToFloat();
		directionalLight.diffuse.y = (float)it->second["diffuse"][1].ToFloat();
		directionalLight.diffuse.z = (float)it->second["diffuse"][2].ToFloat();

		directionalLight.specular.x = (float)it->second["specular"][0].ToFloat();
		directionalLight.specular.y = (float)it->second["specular"][1].ToFloat();
		directionalLight.specular.z = (float)it->second["specular"][2].ToFloat();

		fwrite(&directionalLight, sizeof(DirectionalLight), 1, file);
	}

	// serialize spot lights
	objects = spotLights.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		SpotLight spotLight;

		spotLight.componentId = std::stoi(it->first);
		spotLight.entityId = it->second["entity"].ToInt();

		spotLight.constant = (float)it->second["constant"].ToFloat();
		spotLight.linear = (float)it->second["linear"].ToFloat();
		spotLight.quadratic = (float)it->second["quadratic"].ToFloat();
		spotLight.cutOff = (float)it->second["cutOff"].ToFloat();
		spotLight.outerCutOff = (float)it->second["outerCutOff"].ToFloat();

		spotLight.position.x = (float)it->second["position"][0].ToFloat();
		spotLight.position.y = (float)it->second["position"][1].ToFloat();
		spotLight.position.z = (float)it->second["position"][2].ToFloat();

		spotLight.direction.x = (float)it->second["direction"][0].ToFloat();
		spotLight.direction.y = (float)it->second["direction"][1].ToFloat();
		spotLight.direction.z = (float)it->second["direction"][2].ToFloat();

		spotLight.ambient.x = (float)it->second["ambient"][0].ToFloat();
		spotLight.ambient.y = (float)it->second["ambient"][1].ToFloat();
		spotLight.ambient.z = (float)it->second["ambient"][2].ToFloat();

		spotLight.diffuse.x = (float)it->second["diffuse"][0].ToFloat();
		spotLight.diffuse.y = (float)it->second["diffuse"][1].ToFloat();
		spotLight.diffuse.z = (float)it->second["diffuse"][2].ToFloat();

		spotLight.specular.x = (float)it->second["specular"][0].ToFloat();
		spotLight.specular.y = (float)it->second["specular"][1].ToFloat();
		spotLight.specular.z = (float)it->second["specular"][2].ToFloat();

		fwrite(&spotLight, sizeof(SpotLight), 1, file);
	}

	// serialize point lights
	objects = pointLights.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		PointLight pointLight;

		pointLight.componentId = std::stoi(it->first);
		pointLight.entityId = it->second["entity"].ToInt();

		pointLight.constant = (float)it->second["constant"].ToFloat();
		pointLight.linear = (float)it->second["linear"].ToFloat();
		pointLight.quadratic = (float)it->second["quadratic"].ToFloat();

		pointLight.position.x = (float)it->second["position"][0].ToFloat();
		pointLight.position.y = (float)it->second["position"][1].ToFloat();
		pointLight.position.z = (float)it->second["position"][2].ToFloat();

		pointLight.ambient.x = (float)it->second["ambient"][0].ToFloat();
		pointLight.ambient.y = (float)it->second["ambient"][1].ToFloat();
		pointLight.ambient.z = (float)it->second["ambient"][2].ToFloat();

		pointLight.diffuse.x = (float)it->second["diffuse"][0].ToFloat();
		pointLight.diffuse.y = (float)it->second["diffuse"][1].ToFloat();
		pointLight.diffuse.z = (float)it->second["diffuse"][2].ToFloat();

		pointLight.specular.x = (float)it->second["specular"][0].ToFloat();
		pointLight.specular.y = (float)it->second["specular"][1].ToFloat();
		pointLight.specular.z = (float)it->second["specular"][2].ToFloat();

		fwrite(&pointLight, sizeof(PointLight), 1, file);
	}

	// close file
	if(file){
		fclose(file);
	}



	// verfiy serialization by de-serializing binary file
	SceneHeader sceneHeader = {};
	FILE* file2 = fopen(outputPath.c_str(), "rb");
	if (file2){
		size_t test = fread(&sceneHeader, sizeof(SceneHeader), 1, file2);
		std::cout << "number of bytes read from file: " << test << std::endl;
	}
	else{
		std::cout << "Failed to open file " << outputPath << " for reading" << std::endl;
		return 0;
	}

	std::cout << "de-serialized scene header file contains the following information: " << std::endl;
	std::cout << "fileSize: " << sceneHeader.fileSize << std::endl;
	std::cout << "numberOfEntities: " << sceneHeader.numberOfEntities << std::endl;
	std::cout << "numberOfTransforms: " << sceneHeader.numberOfTransforms << std::endl;
	std::cout << "numberOfMeshRenderers: " << sceneHeader.numberOfMeshRenderers << std::endl;
	std::cout << "numberOfDirectionalLights: " << sceneHeader.numberOfDirectionalLights << std::endl;
	std::cout << "sizeOfEntity: " << sceneHeader.sizeOfEntity << std::endl;
	std::cout << "sizeOfTransform: " << sceneHeader.sizeOfTransform << std::endl;
	std::cout << "sizeOfMeshRenderer: " << sceneHeader.sizeOfMeshRenderer << std::endl;
	std::cout << "sizeOfDirectionalLight: " << sceneHeader.sizeOfDirectionalLight << std::endl;

	if(file2){
		fclose(file2);
	}

	return 1;
}

int serializeMaterials(std::string materialPath)
{
	std::string outputPath = materialPath.substr(0, materialPath.find_last_of(".")) + ".mat";

	return 1;
}

int serializeMeshes(std::string meshPath)
{
	std::string outputPath = meshPath.substr(0, meshPath.find_last_of(".")) + ".mesh";

	return 1;
}

int serializeGMeshes(std::string gmeshPath)
{
	std::string outputPath = gmeshPath.substr(0, gmeshPath.find_last_of(".")) + ".gmesh";

	return 1;
}