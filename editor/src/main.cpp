#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <Windows.h>

#define GLM_FORCE_RADIANS

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <core/Manager.h>
#include <core/Mesh.h>
#include <core/GMesh.h>
#include <core/Entity.h>
#include <core/Guid.h>

#include <components/Transform.h>
#include <components/Rigidbody.h>
#include <components/Camera.h>
#include <components/MeshRenderer.h>
#include <components/LineRenderer.h>
#include <components/DirectionalLight.h>
#include <components/SpotLight.h>
#include <components/PointLight.h>
#include <components/BoxCollider.h>
#include <components/SphereCollider.h>
#include <components/CapsuleCollider.h>

#include <systems/PhysicsSystem.h>
#include <systems/RenderSystem.h>

#include <json/json.hpp>

#include "../../sample_project/include/systems/LogicSystem.h"

#include "../include/MeshLoader.h"

using namespace json;
using namespace PhysicsEngine;


int serializeScene(std::string scenePath);
int serializeMaterials(std::vector<std::string> materialFilePaths);
int serializeMeshes(std::vector<std::string> meshFilePaths);
int serializeGMeshes(std::vector<std::string> gmeshFilePaths);

std::vector<std::string> get_all_files_names_within_folder(std::string folder);

int main(int argc, char* argv[])
{
	// relative path from editor to project directory
	std::string projectDirectory = "../../sample_project/";

	//if(!serializeScene(projectDirectory + "data/scenes/drawcall.json")){ std::cout << "Failed to serialize scene: drawcall.json" << std::endl; }
	if(!serializeScene(projectDirectory + "data/scenes/simple.json")){ std::cout << "Failed to serialize scene: simple.json" << std::endl; }
	//if(!serializeScene(projectDirectory + "data/scenes/pointlight.json")){ std::cout << "Failed to serialize scene: pointlight.json" << std::endl; }
	//if(!serializeScene(projectDirectory + "data/scenes/empty.json")){ std::cout << "Failed to serialize scene: empty.json" << std::endl; }

	// material files
	std::vector<std::string> materialFolderFiles = get_all_files_names_within_folder(projectDirectory + "data/materials");
	std::vector<std::string> materialFilePaths;
	for(unsigned int i = 0; i < materialFolderFiles.size(); i++){
		if(materialFolderFiles[i].substr(materialFolderFiles[i].find_last_of(".") + 1) == "json") {
			materialFilePaths.push_back(projectDirectory + "data/materials/" + materialFolderFiles[i]);
		}
		else
		{
			std::cout << "invalid file: " << materialFolderFiles[i] << std::endl;
		}
	}

	if(!serializeMaterials(materialFilePaths)){
		std::cout << "Failed to serialize materials" << std::endl;
	}

	// mesh files
	std::vector<std::string> meshFolderFiles = get_all_files_names_within_folder(projectDirectory + "data/meshes");
	std::vector<std::string> meshFilePaths;
	for(unsigned int i = 0; i < meshFolderFiles.size(); i++){
		if(meshFolderFiles[i].substr(meshFolderFiles[i].find_last_of(".") + 1) == "json") {
			meshFilePaths.push_back(projectDirectory + "data/meshes/" + meshFolderFiles[i]);
		}
		else
		{
			std::cout << "invalid file: " << meshFolderFiles[i] << std::endl;
		}
	}

	std::cout << "AAAAAAAAAAAAAAAAAAAAAA" << std::endl;

	if(!serializeMeshes(meshFilePaths)){
		std::cout << "Failed to serialize meshes" << std::endl;
	}

	std::cout << "BBBBBBBBBBBBBBBBBBBBBBB" << std::endl;

	// gmesh files
	std::vector<std::string> gmeshFolderFiles = get_all_files_names_within_folder(projectDirectory + "data/gmeshes");
	std::vector<std::string> gmeshFilePaths;
	for(unsigned int i = 0; i < gmeshFolderFiles.size(); i++){
		if(gmeshFolderFiles[i].substr(gmeshFolderFiles[i].find_last_of(".") + 1) == "json") {
			gmeshFilePaths.push_back(projectDirectory + "data/gmeshes/" + gmeshFolderFiles[i]);
		}
		else
		{
			std::cout << "invalid file: " << gmeshFolderFiles[i] << std::endl;
		}
	}

	std::cout << "CCCCCCCCCCCCCCCCCCCCC" << std::endl;

	if(!serializeGMeshes(gmeshFilePaths)){
		std::cout << "Failed to serialize gmeshes" << std::endl;
	}

	std::cout << "DDDDDDDDDDDDDDDDDDDDD" << std::endl;

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
	json::JSON jsonScene = json::JSON::Load(jsonString);

	std::cout << "ABCD" <<std::endl;

	// parse loaded json file
	json::JSON entities;
	json::JSON transforms;
	json::JSON rigidbodies;
	json::JSON cameras;
	json::JSON meshRenderers;
	json::JSON lineRenderers;
	json::JSON directionalLights;
	json::JSON spotLights;
	json::JSON pointLights;
	json::JSON boxColliders;
	json::JSON sphereColliders;
	json::JSON capsuleColliders;
	json::JSON systems;

	json::JSON::JSONWrapper<map<string,JSON>> objects = jsonScene.ObjectRange();
	map<string,JSON>::iterator it;

	size_t sizeOfAllSystems = 0;
	for(it = objects.begin(); it != objects.end(); it++){
		if(it->first == "id"){
			std::cout << "scene id found " << (it->second).ToInt() << std::endl;
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
		else if(type == "Camera"){
			cameras[it->first] = it->second;
		}
		else if(type == "MeshRenderer"){
			meshRenderers[it->first] = it->second;
		}
		else if(type == "LineRenderer"){
			lineRenderers[it->first] = it->second;
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
		else if(type == "BoxCollider"){
			boxColliders[it->first] = it->second;
		}
		else if(type == "SphereCollider"){
			sphereColliders[it->first] = it->second;
		}
		else if(type == "CapsuleCollider"){
			capsuleColliders[it->first] = it->second;
		}
		else if(type == "PhysicsSystem"){
			std::cout << it->first << " is a PhysicsSystem" << std::endl;
			systems[it->first] = it->second;
		}
		else if(type == "RenderSystem"){
			std::cout << it->first << " is a RenderSystem" << std::endl;
			systems[it->first] = it->second;
		}
		else if(type == "PlayerSystem"){
			std::cout << it->first << " is a PlayerSystem" << std::endl;
			systems[it->first] = it->second;
		}
		else if(type == "DebugSystem"){
			std::cout << it->first << " is a DebugSystem" << std::endl;
			systems[it->first] = it->second;
		}
		else if(type == "LogicSystem"){
			std::cout << it->first << " is a LogicSystem" << std::endl;
			systems[it->first] = it->second;
		}
	}

	unsigned int numberOfEntities = std::max(0, entities.size());
	unsigned int numberOfTransforms = std::max(0, transforms.size());
	unsigned int numberOfRigidbodies = std::max(0, rigidbodies.size());
	unsigned int numberOfCameras = std::max(0, cameras.size());
	unsigned int numberOfMeshRenderers = std::max(0, meshRenderers.size());
	unsigned int numberOfLineRenderers = std::max(0, lineRenderers.size());
	unsigned int numberOfDirectionalLights = std::max(0, directionalLights.size());
	unsigned int numberOfSpotLights = std::max(0, spotLights.size());
	unsigned int numberOfPointLights = std::max(0, pointLights.size());
	unsigned int numberOfBoxColliders = std::max(0, boxColliders.size());
	unsigned int numberOfSphereColliders = std::max(0, sphereColliders.size());
	unsigned int numberOfCapsuleColliders = std::max(0, capsuleColliders.size());
	unsigned int numberOfSystems = std::max(0, systems.size());

	std::cout << "number of entities found: " << numberOfEntities << std::endl;
	std::cout << "number of transforms found: " << numberOfTransforms << std::endl;
	std::cout << "number of rigidbodies found: " << numberOfRigidbodies << std::endl;
	std::cout << "number of cameras found" << numberOfCameras << std::endl;
	std::cout << "number of mesh renderers found: " << numberOfMeshRenderers << std::endl;
	std::cout << "number of line renderers found: " << numberOfLineRenderers << std::endl;
	std::cout << "number of directional lights found: " << numberOfDirectionalLights << std::endl;
	std::cout << "number of spot lights found: " << numberOfSpotLights << std::endl;
	std::cout << "number of point lights found: " << numberOfPointLights << std::endl;
	std::cout << "number of box collider found: " << numberOfBoxColliders << std::endl;
	std::cout << "number of sphere collider found: " << numberOfSphereColliders << std::endl;
	std::cout << "number of capsule collider found: " << numberOfCapsuleColliders << std::endl;
	std::cout << "number of systems found: " << numberOfSystems << std::endl;

	// create scene header
	SceneHeader header = {};

	header.numberOfEntities = numberOfEntities;
	header.numberOfTransforms = numberOfTransforms;
	header.numberOfRigidbodies = numberOfRigidbodies;
	header.numberOfCameras = numberOfCameras;
	header.numberOfMeshRenderers = numberOfMeshRenderers;
	header.numberOfLineRenderers = numberOfLineRenderers;
	header.numberOfDirectionalLights = numberOfDirectionalLights;
	header.numberOfSpotLights = numberOfSpotLights;
	header.numberOfPointLights = numberOfPointLights;
	header.numberOfBoxColliders = numberOfBoxColliders;
	header.numberOfSphereColliders = numberOfSphereColliders;
	header.numberOfCapsuleColliders = numberOfCapsuleColliders;
	header.numberOfSystems = numberOfSystems;

	header.sizeOfEntity = sizeof(Entity);
	header.sizeOfTransform = sizeof(Transform);
	header.sizeOfRigidbody = sizeof(Rigidbody);
	header.sizeOfCamera = sizeof(Camera);
	header.sizeOfMeshRenderer = sizeof(MeshRenderer);
	header.sizeOfLineRenderer = sizeof(LineRenderer);
	header.sizeOfDirectionalLight = sizeof(DirectionalLight);
	header.sizeOfSpotLight = sizeof(SpotLight);
	header.sizeOfPointLight = sizeof(PointLight);
	header.sizeOfBoxCollider = sizeof(BoxCollider);
	header.sizeOfSphereCollider = sizeof(SphereCollider);
	header.sizeOfCapsuleCollider = sizeof(CapsuleCollider);

	//header.sizeOfAllSystems = sizeOfAllSystems;

	std::cout << "size of physics system: " << sizeof(PhysicsSystem) << " size of render system: " << sizeof(RenderSystem) << " size of logic system: "  << sizeof(LogicSystem) << std::endl;

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

		entity.entityId = Guid(it->first);

		// for(int i = 0; i < it->second["components"].size(); i++){
		// 	entity.componentIds[i] = it->second["components"][i].ToInt();
		// }

		// for(int i = 0; i < 8; i++){
		// 	std::cout << "component types: " << entity.componentTypes[i] << " globalComponentIndices: " << entity.globalComponentIndices[i] << std::endl;
		// }


		fwrite(&entity, sizeof(Entity), 1, file);
	}

	// serialize transforms
	objects = transforms.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		Transform transform;

		transform.componentId = Guid(it->first);
		transform.entityId = Guid(it->second["entity"].ToString());

		transform.position.x = (float)it->second["position"][0].ToFloat();
		transform.position.y = (float)it->second["position"][1].ToFloat();
		transform.position.z = (float)it->second["position"][2].ToFloat();

		transform.rotation.x = (float)it->second["rotation"][0].ToFloat();
		transform.rotation.y = (float)it->second["rotation"][1].ToFloat();
		transform.rotation.z = (float)it->second["rotation"][2].ToFloat();
		transform.rotation.w = (float)it->second["rotation"][3].ToFloat();

		transform.scale.x = (float)it->second["scale"][0].ToFloat();
		transform.scale.y = (float)it->second["scale"][1].ToFloat();
		transform.scale.z = (float)it->second["scale"][2].ToFloat();

		fwrite(&transform, sizeof(Transform), 1, file);
	}

	// serialize rigidbodies
	objects = rigidbodies.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		Rigidbody rigidbody;

		rigidbody.componentId = Guid(it->first);
		rigidbody.entityId = Guid(it->second["entity"].ToString());

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

		rigidbody.centreOfMass.x = (float)it->second["centreOfMass"][0].ToFloat();
		rigidbody.centreOfMass.y = (float)it->second["centreOfMass"][1].ToFloat();
		rigidbody.centreOfMass.z = (float)it->second["centreOfMass"][2].ToFloat();

		rigidbody.inertiaTensor = glm::mat3(1.0f);
		rigidbody.halfVelocity = glm::vec3(0.0f, 0.0f,0.0f);

		fwrite(&rigidbody, sizeof(Rigidbody), 1, file);
	}

	// serialize cameras 
	objects = cameras.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		Camera camera;

		camera.componentId = Guid(it->first);
		camera.entityId = Guid(it->second["entity"].ToString());

		camera.position.x = (float)it->second["position"][0].ToFloat();
		camera.position.y = (float)it->second["position"][1].ToFloat();
		camera.position.z = (float)it->second["position"][2].ToFloat();

		camera.backgroundColor.x = (float)it->second["backgroundColor"][0].ToFloat();
		camera.backgroundColor.y = (float)it->second["backgroundColor"][1].ToFloat();
		camera.backgroundColor.z = (float)it->second["backgroundColor"][2].ToFloat();
		camera.backgroundColor.w = (float)it->second["backgroundColor"][3].ToFloat();

		fwrite(&camera, sizeof(Camera), 1, file);
	}

	// serialize mesh renderers
	objects = meshRenderers.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		MeshRenderer meshRenderer;

		meshRenderer.componentId = Guid(it->first);
		meshRenderer.entityId = Guid(it->second["entity"].ToString());

		meshRenderer.meshId = Guid(it->second["mesh"].ToString());
		meshRenderer.materialId = Guid(it->second["material"].ToString());

		std::cout << "mesh renderer entity id: " << meshRenderer.entityId.toString() << "mesh renderer component id: " << meshRenderer.componentId.toString() << " mesh renderer mesh id: " << meshRenderer.meshId.toString() << std::endl;

		fwrite(&meshRenderer, sizeof(MeshRenderer), 1, file);
	}

	std::cout << "size of mesh renderer: " << sizeof(MeshRenderer) << std::endl;

	// serialize line renderers
	objects = lineRenderers.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		LineRenderer lineRenderer;

		lineRenderer.componentId = Guid(it->first);
		lineRenderer.entityId = Guid(it->second["entity"].ToString());
		lineRenderer.materialId = Guid(it->second["material"].ToString());

		lineRenderer.start.x = (float)it->second["start"][0].ToFloat();
		lineRenderer.start.y = (float)it->second["start"][1].ToFloat();
		lineRenderer.start.z = (float)it->second["start"][2].ToFloat();

		lineRenderer.end.x = (float)it->second["end"][0].ToFloat();
		lineRenderer.end.y = (float)it->second["end"][1].ToFloat();
		lineRenderer.end.z = (float)it->second["end"][2].ToFloat();

		// lineRenderer.color.x = (float)it->second["color"][0].ToFloat();
		// lineRenderer.color.y = (float)it->second["color"][1].ToFloat();
		// lineRenderer.color.z = (float)it->second["color"][2].ToFloat();
		// lineRenderer.color.w = (float)it->second["color"][3].ToFloat();

		std::cout << "line renderer entity id: " << lineRenderer.entityId.toString() << "line renderer component id: " << lineRenderer.componentId.toString() << std::endl;

		fwrite(&lineRenderer, sizeof(LineRenderer), 1, file);
	}

	std::cout << "size of line renderer: " << sizeof(LineRenderer) << std::endl;

	// serialize directional lights
	objects = directionalLights.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		DirectionalLight directionalLight;

		directionalLight.componentId = Guid(it->first);
		directionalLight.entityId = Guid(it->second["entity"].ToString());

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

		spotLight.componentId = Guid(it->first);
		spotLight.entityId = Guid(it->second["entity"].ToString());

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

		spotLight.projection = glm::perspective(glm::radians(45.0f), 1.0f * 640 / 480, 0.1f, 100.0f);

		fwrite(&spotLight, sizeof(SpotLight), 1, file);
	}

	// serialize point lights
	objects = pointLights.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		PointLight pointLight;

		pointLight.componentId = Guid(it->first);
		pointLight.entityId = Guid(it->second["entity"].ToString());

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

		pointLight.projection = glm::perspective(glm::radians(45.0f), 1.0f * 640 / 480, 0.1f, 100.0f);

		fwrite(&pointLight, sizeof(PointLight), 1, file);
	}

	// serialize box collider
	objects = boxColliders.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		BoxCollider boxCollider;
	}

	// serialize sphere collider
	objects = sphereColliders.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		SphereCollider sphereCollider;

		sphereCollider.componentId = Guid(it->first);
		sphereCollider.entityId = Guid(it->second["entity"].ToString());

		sphereCollider.sphere.centre.x = (float)it->second["centre"][0].ToFloat();
		sphereCollider.sphere.centre.y = (float)it->second["centre"][1].ToFloat();
		sphereCollider.sphere.centre.z = (float)it->second["centre"][2].ToFloat();

		sphereCollider.sphere.radius = (float)it->second["radius"].ToFloat();

		fwrite(&sphereCollider, sizeof(SphereCollider), 1, file);
	}

	// serialize capsule collider
	objects = capsuleColliders.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		CapsuleCollider capsuleCollider;

		capsuleCollider.componentId = Guid(it->first);
		capsuleCollider.entityId = Guid(it->second["entity"].ToString());

		capsuleCollider.capsule.centre.x = (float)it->second["centre"][0].ToFloat();
		capsuleCollider.capsule.centre.x = (float)it->second["centre"][1].ToFloat();
		capsuleCollider.capsule.centre.x = (float)it->second["centre"][2].ToFloat();

		capsuleCollider.capsule.radius = (float)it->second["radius"].ToFloat();
		capsuleCollider.capsule.height = (float)it->second["height"].ToFloat();

		fwrite(&capsuleCollider, sizeof(CapsuleCollider), 1, file);
	}

	// serialize systems;
	objects = systems.ObjectRange();
	for(it = objects.begin(); it != objects.end(); it++){
		size_t systemDataSize;
		int systemType;

		if(it->second["type"].ToString() == "RenderSystem"){
			systemType = 0;
			systemDataSize = sizeof(int);
			fwrite(&systemDataSize, sizeof(size_t), 1, file);
			fwrite(&systemType, systemDataSize, 1, file);
			
			// serialize any other system data here...
		}
		else if(it->second["type"].ToString() == "PhysicsSystem"){
			systemType = 1;
			systemDataSize = sizeof(int);
			fwrite(&systemDataSize, sizeof(size_t), 1, file);
			fwrite(&systemType, systemDataSize, 1, file);

			// serialize any other system data here...

		}
		else if(it->second["type"].ToString() == "DebugSystem"){
			systemType = 3;
			systemDataSize = sizeof(int);
			fwrite(&systemDataSize, sizeof(size_t), 1, file);
			fwrite(&systemType, systemDataSize, 1, file);
			
			// serialize any other system data here...
		}
		else if(it->second["type"].ToString() == "LogicSystem"){
			systemType = 10;
			systemDataSize = sizeof(int);
			fwrite(&systemDataSize, sizeof(size_t), 1, file);
			fwrite(&systemType, systemDataSize, 1, file);
			
			// serialize any other system data here...
		}
		else if(it->second["type"].ToString() == "PlayerSystem"){
			systemType = 11;
			systemDataSize = sizeof(int);
			fwrite(&systemDataSize, sizeof(size_t), 1, file);
			fwrite(&systemType, systemDataSize, 1, file);
			
			// serialize any other system data here...
		}
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
	std::cout << "numberOfRigidbodies: " << sceneHeader.numberOfRigidbodies << std::endl;
	std::cout << "numberOfMeshRenderers: " << sceneHeader.numberOfMeshRenderers << std::endl;
	std::cout << "numberOfLineRenderers: " << sceneHeader.numberOfLineRenderers << std::endl;
	std::cout << "numberOfDirectionalLights: " << sceneHeader.numberOfDirectionalLights << std::endl;
	std::cout << "numberOfSpotLights: " << sceneHeader.numberOfSpotLights << std::endl;
	std::cout << "numberOfPointLights: " << sceneHeader.numberOfPointLights << std::endl;
	std::cout << "numberOfBoxColliders: " << sceneHeader.numberOfBoxColliders << std::endl;
	std::cout << "numberOfSphereColliders: " << sceneHeader.numberOfSphereColliders << std::endl;
	std::cout << "numberOfCapsuleColliders: " << sceneHeader.numberOfCapsuleColliders << std::endl;

	std::cout << "sizeOfEntity: " << sceneHeader.sizeOfEntity << std::endl;
	std::cout << "sizeOfTransform: " << sceneHeader.sizeOfTransform << std::endl;
	std::cout << "sizeOfRigidbody: " << sceneHeader.sizeOfRigidbody << std::endl;
	std::cout << "sizeOfCamera: " << sceneHeader.sizeOfCamera << std::endl;
	std::cout << "sizeOfMeshRenderer: " << sceneHeader.sizeOfMeshRenderer << std::endl;
	std::cout << "sizeOfLineRenderer: " << sceneHeader.sizeOfLineRenderer << std::endl;
	std::cout << "sizeOfDirectionalLight: " << sceneHeader.sizeOfDirectionalLight << std::endl;
	std::cout << "sizeOfSpotLight: " << sceneHeader.sizeOfSpotLight << std::endl;
	std::cout << "sizeOfPointLight: " << sceneHeader.sizeOfPointLight << std::endl;
	std::cout << "sizeOfBoxCollider: " << sceneHeader.sizeOfBoxCollider << std::endl;
	std::cout << "sizeOfSphereCollider: " << sceneHeader.sizeOfSphereCollider << std::endl;
	std::cout << "sizeOfCapsuleCollider: " << sceneHeader.sizeOfCapsuleCollider << std::endl;

	if(file2){
		fclose(file2);
	}

	return 1;
}

int serializeMaterials(std::vector<std::string> materialFilePaths)
{
	for(unsigned int i = 0; i < materialFilePaths.size(); i++){

		// open json file and load to json object
		std::ifstream in(materialFilePaths[i], std::ios::in | std::ios::binary);
		std::ostringstream contents;
		contents << in.rdbuf();
		in.close();

		std::string jsonString = contents.str();
		json::JSON jsonMaterial = JSON::Load(jsonString);

		Material material;
		material.assetId = Guid(jsonMaterial["id"].ToString());
		material.shininess = (float)jsonMaterial["shininess"].ToFloat();
		material.ambient.x = (float)jsonMaterial["ambient"][0].ToFloat();
		material.ambient.y = (float)jsonMaterial["ambient"][1].ToFloat();
		material.ambient.z = (float)jsonMaterial["ambient"][2].ToFloat();
		material.diffuse.x = (float)jsonMaterial["diffuse"][0].ToFloat();
		material.diffuse.y = (float)jsonMaterial["diffuse"][1].ToFloat();
		material.diffuse.z = (float)jsonMaterial["diffuse"][2].ToFloat();
		material.specular.x = (float)jsonMaterial["specular"][0].ToFloat();
		material.specular.y = (float)jsonMaterial["specular"][1].ToFloat();
		material.specular.z = (float)jsonMaterial["specular"][2].ToFloat();
		material.shaderId  = Guid(jsonMaterial["shader"].ToString());
		material.textureId = Guid(jsonMaterial["mainTexture"].ToString());
		material.normalMapId = Guid(jsonMaterial["normalMap"].ToString());
		material.specularMapId = Guid(jsonMaterial["specularMap"].ToString());

		std::string outputPath = materialFilePaths[i].substr(0, materialFilePaths[i].find_last_of(".")) + ".mat";

		std::cout << "outputPath: " << outputPath << " material id: " << material.assetId.toString() << " shader id: " << material.shaderId.toString() << " main texture id: " << material.textureId.toString() << std::endl;

		// serialize material
		FILE* file = fopen(outputPath.c_str(), "wb");
		if (file){
			size_t test = fwrite(&material, sizeof(Material), 1, file);
			std::cout << "number of bytes written to file: " << test << std::endl;
		}
		else{
			std::cout << "Failed to open file " << outputPath << " for writing." << std::endl;
			return 0;
		}

		if(file){
			fclose(file);
		}
	}

	return 1;
}

int serializeMeshes(std::vector<std::string> meshFilePaths)
{
	for(unsigned int i = 0 ; i < meshFilePaths.size(); i++){
		// open json file and load to json object
		std::ifstream in(meshFilePaths[i], std::ios::in | std::ios::binary);
		std::ostringstream contents;
		contents << in.rdbuf();
		in.close();

		std::string jsonString = contents.str();
		json::JSON jsonMesh = JSON::Load(jsonString);

		Mesh mesh;

		std::string filePath = meshFilePaths[i].substr(0, meshFilePaths[i].find_last_of(".")) + ".txt";

		std::cout << "mesh filepath: " << filePath << std::endl;

		if(MeshLoader::load(filePath, mesh)){
			
			// create mesh header
			MeshHeader header = {};
			header.meshId = Guid(jsonMesh["id"].ToString());
			header.verticesSize = (unsigned int)mesh.vertices.size();
			header.normalsSize = (unsigned int)mesh.normals.size();
			header.texCoordsSize = (unsigned int)mesh.texCoords.size();

			std::cout << "vertices size: " << mesh.vertices.size() << " normals size: " << mesh.normals.size() << " texCoords size: " << mesh.texCoords.size() << std::endl;

			std::string outputPath = meshFilePaths[i].substr(0, meshFilePaths[i].find_last_of(".")) + ".mesh";

			// for(unsigned int i = 0; i < mesh.vertices.size(); i++){
			// 	std::cout << mesh.vertices[i] << " ";
			// }
			// std::cout << "" << std::endl;

			// serialize scene header and mesh data
			FILE* file = fopen(outputPath.c_str(), "wb");
			if (file){
				size_t test = fwrite(&header, sizeof(MeshHeader), 1, file);
				test += fwrite(&(mesh.vertices[0]), mesh.vertices.size()*sizeof(float), 1, file);
				test += fwrite(&(mesh.normals[0]), mesh.normals.size()*sizeof(float), 1, file);
				test += fwrite(&(mesh.texCoords[0]), mesh.texCoords.size()*sizeof(float), 1, file);
				std::cout << "number of bytes written to file: " << test << std::endl;
			}
			else{
				std::cout << "Failed to open file " << outputPath << " for writing." << std::endl;
				return 0;
			}

			if(file){
				fclose(file);
			}
		}
		else{
			std::cout << "Failed to open file " << filePath << " for parsing" << std::endl;
			return 0;
		}
	}

	return 1;
}

int serializeGMeshes(std::vector<std::string> gmeshFilePaths)
{
	for(unsigned int i = 0 ; i < gmeshFilePaths.size(); i++){
	// open json file and load to json object
		std::ifstream in(gmeshFilePaths[i], std::ios::in | std::ios::binary);
		std::ostringstream contents;
		contents << in.rdbuf();
		in.close();

		std::string jsonString = contents.str();
		json::JSON jsonGMesh = JSON::Load(jsonString);

		GMesh gmesh;

		std::string filePath = gmeshFilePaths[i].substr(0, gmeshFilePaths[i].find_last_of(".")) + ".msh";

		if(MeshLoader::load(filePath, gmesh)){
			
			// create gmesh header
			GMeshHeader header = {};
			header.gmeshId = Guid(jsonGMesh["id"].ToString());
			header.dim = gmesh.dim;
			header.ng = gmesh.ng;
		    header.n = gmesh.n;
		    header.nte = gmesh.nte;
		    header.ne = gmesh.ne;
		    header.ne_b = gmesh.ne_b;
		    header.npe = gmesh.npe;
		    header.npe_b = gmesh.npe_b;
		    header.type = gmesh.type;
		    header.type_b = gmesh.type_b;
			header.verticesSize = (unsigned int)gmesh.vertices.size();
			header.connectSize = (unsigned int)gmesh.connect.size();
			header.bconnectSize = (unsigned int)gmesh.bconnect.size();
			header.groupsSize = (unsigned int)gmesh.groups.size();

			std::cout << "vertices size: " << gmesh.vertices.size() << " connect size: " << gmesh.connect.size() << " bconnect size: " << gmesh.bconnect.size() << " groups size: " << gmesh.groups.size() << std::endl;

			std::string outputPath = gmeshFilePaths[i].substr(0, gmeshFilePaths[i].find_last_of(".")) + ".gmesh";

			// serialize scene header and mesh data
			FILE* file = fopen(outputPath.c_str(), "wb");
			if (file){
				size_t test = fwrite(&header, sizeof(GMeshHeader), 1, file);
				test += fwrite(&gmesh.vertices[0], gmesh.vertices.size()*sizeof(float), 1, file);
				test += fwrite(&gmesh.connect[0], gmesh.connect.size()*sizeof(float), 1, file);
				test += fwrite(&gmesh.bconnect[0], gmesh.bconnect.size()*sizeof(float), 1, file);
				test += fwrite(&gmesh.groups[0], gmesh.groups.size()*sizeof(float), 1, file);
				std::cout << "number of bytes written to file: " << test << std::endl;
			}
			else{
				std::cout << "Failed to open file " << outputPath << " for writing." << std::endl;
				return 0;
			}

			if(file){
				fclose(file);
			}
		}
		else{
			std::cout << "Failed to open file " << filePath << " for parsing" << std::endl;
			return 0;
		}
	}

	return 1;
}

std::vector<std::string> get_all_files_names_within_folder(std::string folder)
{
    std::vector<std::string> names;
    std::string search_path = folder + "/*.*";
    WIN32_FIND_DATA fd; 
    HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd); 
    if(hFind != INVALID_HANDLE_VALUE) { 
        do { 
            // read all (real) files in current folder
            // , delete '!' read other 2 default folder . and ..
            if(! (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ) {
                names.push_back(fd.cFileName);
            }
        }while(::FindNextFile(hFind, &fd)); 
        ::FindClose(hFind); 
    } 
    return names;
}