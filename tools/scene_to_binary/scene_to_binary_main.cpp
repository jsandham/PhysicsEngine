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

#include <core/Load.h>
#include <core/AssetLoader.h>
#include <core/Material.h>
#include <core/Shader.h>
#include <core/Texture2D.h>
#include <core/Mesh.h>
#include <core/GMesh.h>
#include <core/Font.h>
#include <core/Entity.h>
#include <core/Guid.h>

#include <components/Transform.h>
#include <components/Rigidbody.h>
#include <components/Camera.h>
#include <components/MeshRenderer.h>
#include <components/LineRenderer.h>

#include <components/BoxCollider.h>
#include <components/SphereCollider.h>
#include <components/MeshCollider.h>
#include <components/CapsuleCollider.h>
// #include <components/Boids.h>

#include <systems/PhysicsSystem.h>
#include <systems/RenderSystem.h>

#include <json/json.hpp>

#include "../../../sample_project/Demo/Demo/include/LogicSystem.h"
#include "../../../sample_project/Demo/Demo/include/PlayerSystem.h"

using namespace json;
using namespace PhysicsEngine;


int serializeScenes(std::string projectDirectory);
int serializeAssets(std::string projectDirectory);

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

int main(int argc, char* argv[])
{
	// relative path from editor to project directory
	std::string projectDirectory = "../../../sample_project/Demo/Demo/";

	if(!serializeScenes(projectDirectory)){ std::cout << "Failed to serialize scenes" << std::endl; }
	if(!serializeAssets(projectDirectory)){ std::cout << "Failed to serialize assets" << std::endl; }

	while(true){}

	return 0;
}


int serializeScenes(std::string projectDirectory)
{
	// scene files
	std::vector<std::string> sceneFolderFiles = get_all_files_names_within_folder(projectDirectory + "data/scenes");
	std::vector<std::string> sceneFilePaths;
	for(unsigned int i = 0; i < sceneFolderFiles.size(); i++){
		if(sceneFolderFiles[i].substr(sceneFolderFiles[i].find_last_of(".") + 1) == "scene") {
			sceneFilePaths.push_back(projectDirectory + "data/scenes/" + sceneFolderFiles[i]);
		}
		else
		{
			//std::cout << "invalid file: " << sceneFolderFiles[i] << std::endl;
		}
	}

	// loop through all scenes and serialize them
	for(size_t i = 0; i < sceneFilePaths.size(); i++){
		std::string outputPath = sceneFilePaths[i].substr(sceneFilePaths[i].find_last_of("\\/") + 1);
		outputPath = "scenes\\" + outputPath.substr(0, outputPath.find_last_of(".")) + ".data";
		std::cout << "input path: " << sceneFilePaths[i] << " output path: " << outputPath << std::endl;

		// open json file and load to json object
		std::ifstream in(sceneFilePaths[i], std::ios::in | std::ios::binary);
		std::ostringstream contents;
		contents << in.rdbuf();
		in.close();
		std::string jsonString = contents.str();
		json::JSON jsonScene = json::JSON::Load(jsonString);

		// parse loaded json file
		json::JSON entities;
		json::JSON transforms;
		json::JSON rigidbodies;
		json::JSON cameras;
		json::JSON meshRenderers;
		json::JSON lineRenderers;
		json::JSON lights;
		json::JSON boxColliders;
		json::JSON sphereColliders;
		json::JSON meshColliders;
		json::JSON capsuleColliders;
		//json::JSON boids;
		json::JSON systems;

		json::JSON::JSONWrapper<map<string,JSON>> objects = jsonScene.ObjectRange();
		map<string,JSON>::iterator it;

		size_t sizeOfAllSystems = 0;
		for(it = objects.begin(); it != objects.end(); it++){
			std::string type = it->second["type"].ToString();

			if(type == "Entity"){
				//std::cout << it->first << " is an Entity" << std::endl;
				entities[it->first] = it->second;
			}
			else if(type == "Transform"){
				//std::cout << it->first << " is a Transform" << std::endl;
				transforms[it->first] = it->second;
			}
			else if(type == "Rigidbody"){
				//std::cout << it->first << " is a Rigidbody" << std::endl;
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
			else if(type == "Light"){
				lights[it->first] = it->second;
			}
			else if(type == "BoxCollider"){
				boxColliders[it->first] = it->second;
			}
			else if(type == "SphereCollider"){
				sphereColliders[it->first] = it->second;
			}
			else if(type == "MeshCollider"){
				meshColliders[it->first] = it->second;
			}
			else if(type == "CapsuleCollider"){
				capsuleColliders[it->first] = it->second;
			}
			// else if(type == "Boids"){
			// 	boids[it->first] = it->second;
			// }
			else if(type == "PhysicsSystem"){
				//std::cout << it->first << " is a PhysicsSystem" << std::endl;
				systems[it->first] = it->second;
			}
			else if(type == "RenderSystem"){
				//std::cout << it->first << " is a RenderSystem" << std::endl;
				systems[it->first] = it->second;
			}
			else if(type == "CleanUpSystem"){
				//std::cout << it->first << " is a PhysicsSystem" << std::endl;
				systems[it->first] = it->second;
			}
			else if(type == "PlayerSystem"){
				//std::cout << it->first << " is a PlayerSystem" << std::endl;
				systems[it->first] = it->second;
			}
			else if(type == "DebugSystem"){
				//std::cout << it->first << " is a DebugSystem" << std::endl;
				systems[it->first] = it->second;
			}
			// else if(type == "BoidsSystem"){
			// 	//std::cout << it->first << " is a LogicSystem" << std::endl;
			// 	systems[it->first] = it->second;
			// }
			else if(type == "LogicSystem"){
				//std::cout << it->first << " is a LogicSystem" << std::endl;
				systems[it->first] = it->second;
			}
		}

		unsigned int numberOfEntities = std::max(0, entities.size());
		unsigned int numberOfTransforms = std::max(0, transforms.size());
		unsigned int numberOfRigidbodies = std::max(0, rigidbodies.size());
		unsigned int numberOfCameras = std::max(0, cameras.size());
		unsigned int numberOfMeshRenderers = std::max(0, meshRenderers.size());
		unsigned int numberOfLineRenderers = std::max(0, lineRenderers.size());
		unsigned int numberOfLights = std::max(0, lights.size());
		unsigned int numberOfBoxColliders = std::max(0, boxColliders.size());
		unsigned int numberOfSphereColliders = std::max(0, sphereColliders.size());
		unsigned int numberOfMeshColliders = std::max(0, meshColliders.size());
		unsigned int numberOfCapsuleColliders = std::max(0, capsuleColliders.size());
		//unsigned int numberOfBoids = std::max(0, boids.size());
		unsigned int numberOfSystems = std::max(0, systems.size());

		std::cout << "number of entities found: " << numberOfEntities << std::endl;
		std::cout << "number of transforms found: " << numberOfTransforms << std::endl;
		std::cout << "number of rigidbodies found: " << numberOfRigidbodies << std::endl;
		std::cout << "number of cameras found" << numberOfCameras << std::endl;
		std::cout << "number of mesh renderers found: " << numberOfMeshRenderers << std::endl;
		std::cout << "number of line renderers found: " << numberOfLineRenderers << std::endl;
		std::cout << "number of lights found: " << numberOfLights << std::endl;
		std::cout << "number of box collider found: " << numberOfBoxColliders << std::endl;
		std::cout << "number of sphere collider found: " << numberOfSphereColliders << std::endl;
		std::cout << "number of mesh collider found: " << numberOfMeshColliders << std::endl;
		std::cout << "number of capsule collider found: " << numberOfCapsuleColliders << std::endl;
		//std::cout << "number of boids found: " << numberOfBoids << std::endl;
		std::cout << "number of systems found: " << numberOfSystems << std::endl;

		// create scene header
		SceneHeader header = {};

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
		if(!entities.IsNull()){
			json::JSON::JSONWrapper<map<string,JSON>> entityObjects = entities.ObjectRange();
			// map<string,JSON>::iterator it;
			// entityObjects = entities.ObjectRange();
			for(it = entityObjects.begin(); it != entityObjects.end(); it++){
				Entity entity;
				entity.entityId = Guid(it->first);

				std::vector<char> data = entity.serialize();

				char classification = 'e';
				int type = 0;
				size_t size = data.size();

				fwrite(&classification, sizeof(char), 1, file);
				fwrite(&type, sizeof(int), 1, file);
				fwrite(&size, sizeof(size_t), 1, file);
				fwrite(&data[0], data.size(), 1, file);
			}
		}

		// serialize transforms
		if(!transforms.IsNull()){
			json::JSON::JSONWrapper<map<string,JSON>> transformObjects = transforms.ObjectRange();
			for(it = transformObjects.begin(); it != transformObjects.end(); it++){
				Transform transform;

				transform.componentId = Guid(it->first);
				transform.parentId = Guid(it->second["parent"].ToString());
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

				std::vector<char> data = transform.serialize();

				char classification = 'c';
				int type = 0;
				size_t size = data.size();

				fwrite(&classification, sizeof(char), 1, file);
				fwrite(&type, sizeof(int), 1, file);
				fwrite(&size, sizeof(size_t), 1, file);
				fwrite(&data[0], data.size(), 1, file);
			}
		}

		std::cout << "AAAAAA" << std::endl; 

		// serialize rigidbodies
		if(!rigidbodies.IsNull()){
			json::JSON::JSONWrapper<map<string,JSON>> rigidbodyObjects = rigidbodies.ObjectRange();
			if(!rigidbodies.IsNull()){
				for(it = rigidbodyObjects.begin(); it != rigidbodyObjects.end(); it++){
					Rigidbody rigibody;

					rigibody.componentId = Guid(it->first);
					rigibody.entityId = Guid(it->second["entity"].ToString());

					rigibody.useGravity = (bool)it->second["useGravity"].ToBool();
					rigibody.mass = (float)it->second["mass"].ToFloat();
					rigibody.drag = (float)it->second["drag"].ToFloat();
					rigibody.angularDrag = (float)it->second["angularDrag"].ToFloat();

					rigibody.velocity.x = (float)it->second["velocity"][0].ToFloat();
					rigibody.velocity.y = (float)it->second["velocity"][1].ToFloat();
					rigibody.velocity.z = (float)it->second["velocity"][2].ToFloat();

					rigibody.angularVelocity.x = (float)it->second["angularVelocity"][0].ToFloat();
					rigibody.angularVelocity.y = (float)it->second["angularVelocity"][1].ToFloat();
					rigibody.angularVelocity.z = (float)it->second["angularVelocity"][2].ToFloat();

					rigibody.centreOfMass.x = (float)it->second["centreOfMass"][0].ToFloat();
					rigibody.centreOfMass.y = (float)it->second["centreOfMass"][1].ToFloat();
					rigibody.centreOfMass.z = (float)it->second["centreOfMass"][2].ToFloat();

					rigibody.inertiaTensor = glm::mat3(1.0f);
					rigibody.halfVelocity = glm::vec3(0.0f, 0.0f,0.0f);

					std::vector<char> data = rigibody.serialize();

					char classification = 'c';
					int type = 1;
					size_t size = data.size();

					fwrite(&classification, sizeof(char), 1, file);
					fwrite(&type, sizeof(int), 1, file);
					fwrite(&size, sizeof(size_t), 1, file);
					fwrite(&data[0], data.size(), 1, file);
				}
			}
		}

		// serialize cameras 
		if(!cameras.IsNull()){
			json::JSON::JSONWrapper<map<string,JSON>> cameraObjects = cameras.ObjectRange();
			for(it = cameraObjects.begin(); it != cameraObjects.end(); it++){
				Camera camera;

				camera.componentId = Guid(it->first);
				camera.entityId = Guid(it->second["entity"].ToString());

				camera.position.x = (float)it->second["position"][0].ToFloat();
				camera.position.y = (float)it->second["position"][1].ToFloat();
				camera.position.z = (float)it->second["position"][2].ToFloat();

				camera.front.x = (float)it->second["front"][0].ToFloat();
				camera.front.y = (float)it->second["front"][1].ToFloat();
				camera.front.z = (float)it->second["front"][2].ToFloat();

				camera.up.x = (float)it->second["up"][0].ToFloat();
				camera.up.y = (float)it->second["up"][1].ToFloat();
				camera.up.z = (float)it->second["up"][2].ToFloat();

				camera.backgroundColor.x = (float)it->second["backgroundColor"][0].ToFloat();
				camera.backgroundColor.y = (float)it->second["backgroundColor"][1].ToFloat();
				camera.backgroundColor.z = (float)it->second["backgroundColor"][2].ToFloat();
				camera.backgroundColor.w = (float)it->second["backgroundColor"][3].ToFloat();

				camera.viewport.x = (int)it->second["x"].ToInt();
				camera.viewport.y = (int)it->second["y"].ToInt();
				camera.viewport.width = (int)it->second["width"].ToInt();
				camera.viewport.height = (int)it->second["height"].ToInt();

				camera.frustum.fov = (float)it->second["fov"].ToFloat();
				camera.frustum.nearPlane = (float)it->second["near"].ToFloat();
				camera.frustum.farPlane = (float)it->second["far"].ToFloat();

				std::vector<char> data = camera.serialize();

				char classification = 'c';
				int type = 2;
				size_t size = data.size();

				fwrite(&classification, sizeof(char), 1, file);
				fwrite(&type, sizeof(int), 1, file);
				fwrite(&size, sizeof(size_t), 1, file);
				fwrite(&data[0], data.size(), 1, file);
			}
		}

		std::cout << "CCCCCCCCC" << std::endl; 

		// serialize mesh renderers
		if(!meshRenderers.IsNull()){
			json::JSON::JSONWrapper<map<string,JSON>> meshRendererObjects = meshRenderers.ObjectRange();
			for(it = meshRendererObjects.begin(); it != meshRendererObjects.end(); it++){
				MeshRenderer meshRenderer;

				meshRenderer.componentId = Guid(it->first);
				meshRenderer.entityId = Guid(it->second["entity"].ToString());
				meshRenderer.meshId = Guid(it->second["mesh"].ToString());

				if(it->second.hasKey("material")){
					meshRenderer.materialIds[0] = Guid(it->second["material"].ToString());
					for(int j = 1; j < 8; j++){
						meshRenderer.materialIds[j] = Guid::INVALID;
					}
				}
				else if(it->second.hasKey("materials")){
					int materialCount = it->second["materials"].length();
					if(materialCount > 8){
						std::cout << "Error: Currently only support at most 8 materials" << std::endl;
						return 0;
					}

					for(int j = 0; j < materialCount; j++){
						meshRenderer.materialIds[j] = Guid(it->second["materials"][j].ToString());
					}

					for(int j = materialCount; j < 8; j++){
						meshRenderer.materialIds[j] = Guid::INVALID;
					}
				}

				meshRenderer.isStatic = it->second["isStatic"].ToBool();

				std::vector<char> data = meshRenderer.serialize();

				char classification = 'c';
				int type = 3;
				size_t size = data.size();

				fwrite(&classification, sizeof(char), 1, file);
				fwrite(&type, sizeof(int), 1, file);
				fwrite(&size, sizeof(size_t), 1, file);
				fwrite(&data[0], data.size(), 1, file);
			}
		}

		std::cout << "size of mesh renderer: " << sizeof(MeshRendererHeader) << std::endl;

		// serialize line renderers
		if(!lineRenderers.IsNull()){
			json::JSON::JSONWrapper<map<string,JSON>> lineRendererObjects = lineRenderers.ObjectRange();
			for(it = lineRendererObjects.begin(); it != lineRendererObjects.end(); it++){
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

				std::vector<char> data = lineRenderer.serialize();

				char classification = 'c';
				int type = 4;
				size_t size = data.size();

				fwrite(&classification, sizeof(char), 1, file);
				fwrite(&type, sizeof(int), 1, file);
				fwrite(&size, sizeof(size_t), 1, file);
				fwrite(&data[0], data.size(), 1, file);
			}
		}

		std::cout << "size of line renderer: " << sizeof(LineRenderer) << std::endl;

		// serialize lights
		if(!lights.IsNull()){
			json::JSON::JSONWrapper<map<string,JSON>> lightObjects = lights.ObjectRange();
			for(it = lightObjects.begin(); it != lightObjects.end(); it++){
				Light light;

				light.componentId = Guid(it->first);
				light.entityId = Guid(it->second["entity"].ToString());

				light.position.x = (float)it->second["position"][0].ToFloat();
				light.position.y = (float)it->second["position"][1].ToFloat();
				light.position.z = (float)it->second["position"][2].ToFloat();

				light.direction.x = (float)it->second["direction"][0].ToFloat();
				light.direction.y = (float)it->second["direction"][1].ToFloat();
				light.direction.z = (float)it->second["direction"][2].ToFloat();

				light.ambient.x = (float)it->second["ambient"][0].ToFloat();
				light.ambient.y = (float)it->second["ambient"][1].ToFloat();
				light.ambient.z = (float)it->second["ambient"][2].ToFloat();

				light.diffuse.x = (float)it->second["diffuse"][0].ToFloat();
				light.diffuse.y = (float)it->second["diffuse"][1].ToFloat();
				light.diffuse.z = (float)it->second["diffuse"][2].ToFloat();

				light.specular.x = (float)it->second["specular"][0].ToFloat();
				light.specular.y = (float)it->second["specular"][1].ToFloat();
				light.specular.z = (float)it->second["specular"][2].ToFloat();

				light.constant = (float)it->second["constant"].ToFloat();
				light.linear = (float)it->second["linear"].ToFloat();
				light.quadratic = (float)it->second["quadratic"].ToFloat();
				light.cutOff = (float)it->second["cutOff"].ToFloat();
				light.outerCutOff = (float)it->second["outerCutOff"].ToFloat();

				light.lightType = static_cast<LightType>((int)it->second["lightType"].ToInt());
				light.shadowType = static_cast<ShadowType>((int)it->second["shadowType"].ToInt());

				std::cout << "Light type: " << (int)light.lightType << " shadow type: " << (int)light.shadowType << std::endl;

				light.projection = glm::perspective(2.0f * glm::radians(light.outerCutOff), 1.0f * 1024 / 1024, 0.1f, 12.0f);

				std::vector<char> data = light.serialize();

				char classification = 'c';
				int type = 5;
				size_t size = data.size();

				fwrite(&classification, sizeof(char), 1, file);
				fwrite(&type, sizeof(int), 1, file);
				fwrite(&size, sizeof(size_t), 1, file);
				fwrite(&data[0], data.size(), 1, file);
			}
		}

		// serialize box collider
		if(!boxColliders.IsNull()){
			json::JSON::JSONWrapper<map<string,JSON>> boxColliderObjects = boxColliders.ObjectRange();
			for(it = boxColliderObjects.begin(); it != boxColliderObjects.end(); it++){
				BoxCollider collider;

				collider.componentId = Guid(it->first);
				collider.entityId = Guid(it->second["entity"].ToString());

				collider.bounds.centre.x = (float)it->second["centre"][0].ToFloat();
				collider.bounds.centre.y = (float)it->second["centre"][1].ToFloat();
				collider.bounds.centre.z = (float)it->second["centre"][2].ToFloat();

				collider.bounds.size.x = (float)it->second["size"][0].ToFloat();
				collider.bounds.size.y = (float)it->second["size"][1].ToFloat();
				collider.bounds.size.z = (float)it->second["size"][2].ToFloat();

				std::vector<char> data = collider.serialize();

				char classification = 'c';
				int type = 8;
				size_t size = data.size();

				fwrite(&classification, sizeof(char), 1, file);
				fwrite(&type, sizeof(int), 1, file);
				fwrite(&size, sizeof(size_t), 1, file);
				fwrite(&data[0], data.size(), 1, file);
			}
		}

		// serialize sphere collider
		if(!sphereColliders.IsNull()){
			json::JSON::JSONWrapper<map<string,JSON>> sphereColliderObjects = sphereColliders.ObjectRange();
			// map<string,JSON>::iterator it;
			// objects = sphereColliders.ObjectRange();
			for(it = sphereColliderObjects.begin(); it != sphereColliderObjects.end(); it++){
				SphereCollider collider;

				collider.componentId = Guid(it->first);
				collider.entityId = Guid(it->second["entity"].ToString());

				collider.sphere.centre.x = (float)it->second["centre"][0].ToFloat();
				collider.sphere.centre.y = (float)it->second["centre"][1].ToFloat();
				collider.sphere.centre.z = (float)it->second["centre"][2].ToFloat();

				collider.sphere.radius = (float)it->second["radius"].ToFloat();

				std::vector<char> data = collider.serialize();

				char classification = 'c';
				int type = 9;
				size_t size = data.size();

				fwrite(&classification, sizeof(char), 1, file);
				fwrite(&type, sizeof(int), 1, file);
				fwrite(&size, sizeof(size_t), 1, file);
				fwrite(&data[0], data.size(), 1, file);
			}
		}

		// serialize mesh collider
		if(!meshColliders.IsNull()){
			json::JSON::JSONWrapper<map<string,JSON>> meshColliderObjects = meshColliders.ObjectRange();
			for(it = meshColliderObjects.begin(); it != meshColliderObjects.end(); it++){
				MeshCollider collider;

				collider.componentId = Guid(it->first);
				collider.entityId = Guid(it->second["entity"].ToString());
				collider.meshId = Guid(it->second["mesh"].ToString());

				std::vector<char> data = collider.serialize();

				char classification = 'c';
				int type = 15;
				size_t size = data.size();

				fwrite(&classification, sizeof(char), 1, file);
				fwrite(&type, sizeof(int), 1, file);
				fwrite(&size, sizeof(size_t), 1, file);
				fwrite(&data[0], data.size(), 1, file);
			}
		}

		// serialize capsule collider
		if(!capsuleColliders.IsNull()){
			json::JSON::JSONWrapper<map<string,JSON>> capsuleColliderObjects = capsuleColliders.ObjectRange();
			for(it = capsuleColliderObjects.begin(); it != capsuleColliderObjects.end(); it++){
				CapsuleCollider collider;

				collider.componentId = Guid(it->first);
				collider.entityId = Guid(it->second["entity"].ToString());

				collider.capsule.centre.x = (float)it->second["centre"][0].ToFloat();
				collider.capsule.centre.x = (float)it->second["centre"][1].ToFloat();
				collider.capsule.centre.x = (float)it->second["centre"][2].ToFloat();

				collider.capsule.radius = (float)it->second["radius"].ToFloat();
				collider.capsule.height = (float)it->second["height"].ToFloat();

				std::vector<char> data = collider.serialize();

				char classification = 'c';
				int type = 10;
				size_t size = data.size();

				fwrite(&classification, sizeof(char), 1, file);
				fwrite(&type, sizeof(int), 1, file);
				fwrite(&size, sizeof(size_t), 1, file);
				fwrite(&data[0], data.size(), 1, file);
			}
		}

		// serialize boids
		// if(!boids.IsNull()){
		// 	json::JSON::JSONWrapper<map<string,JSON>> boidsObjects = boids.ObjectRange();
		// 	for(it = boidsObjects.begin(); it != boidsObjects.end(); it++){
		// 		Boids boid;

		// 		boid.componentId = Guid(it->first);
		// 		boid.entityId = Guid(it->second["entity"].ToString());
		// 		boid.meshId = Guid(it->second["mesh"].ToString());
		// 		boid.shaderId = Guid(it->second["shader"].ToString());

		// 		boid.numBoids = it->second["numBoids"].ToInt();
		// 		boid.h = (float)it->second["h"].ToFloat();
		// 		boid.bounds.centre.x = (float)it->second["centre"][0].ToFloat();
		// 		boid.bounds.centre.y = (float)it->second["centre"][1].ToFloat();
		// 		boid.bounds.centre.z = (float)it->second["centre"][2].ToFloat();

		// 		boid.bounds.size.x = (float)it->second["size"][0].ToFloat();
		// 		boid.bounds.size.y = (float)it->second["size"][1].ToFloat();
		// 		boid.bounds.size.z = (float)it->second["size"][2].ToFloat();

		// 		std::vector<char> data = boid.serialize();

		// 		char classification = 'c';
		// 		int type = 11;
		// 		size_t size = data.size();

		// 		fwrite(&classification, sizeof(char), 1, file);
		// 		fwrite(&type, sizeof(int), 1, file);
		// 		fwrite(&size, sizeof(size_t), 1, file);
		// 		fwrite(&data[0], data.size(), 1, file);
		// 	}
		// }

		// serialize systems;
		if(!systems.IsNull()){
			json::JSON::JSONWrapper<map<string,JSON>> systemObjects = systems.ObjectRange();
			for(it = systemObjects.begin(); it != systemObjects.end(); it++){
				char classification = 's';
				int type;
				size_t size;

				std::cout << "System type: " << it->second["type"].ToString() << std::endl;

				if(it->second["type"].ToString() == "RenderSystem"){
					type = 0;
				}
				else if(it->second["type"].ToString() == "PhysicsSystem"){
					type = 1;
				}
				else if(it->second["type"].ToString() == "CleanUpSystem"){
					type = 2;
				}
				else if(it->second["type"].ToString() == "DebugSystem"){
					type = 3;
				}
				// else if(it->second["type"].ToString() == "BoidsSystem"){
				// 	type = 4;
				// }
				else if(it->second["type"].ToString() == "LogicSystem"){
					type = 20;
				}
				else if(it->second["type"].ToString() == "PlayerSystem"){
					type = 21;
				}

				int order = it->second["order"].ToInt();

				std::vector<char> data(4);
				memcpy(&data[0], &order, sizeof(int));

				size = data.size();

				fwrite(&classification, sizeof(char), 1, file);
				fwrite(&type, sizeof(int), 1, file);
				fwrite(&size, sizeof(size_t), 1, file);
				fwrite(&data[0], data.size(), 1, file);
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

		if(file2){
			fclose(file2);
		}
	}

	return 1;
}


int serializeAssets(std::string projectDirectory)
{
	std::cout << "Serialize assets" << std::endl;

	// shader files
	std::vector<std::string> shaderFolderFiles = get_all_files_names_within_folder(projectDirectory + "data/shaders");
	std::vector<std::string> shaderFilePaths;
	for(unsigned int i = 0; i < shaderFolderFiles.size(); i++){
		if(shaderFolderFiles[i].substr(shaderFolderFiles[i].find_last_of(".") + 1) == "json") {
			shaderFilePaths.push_back(projectDirectory + "data/shaders/" + shaderFolderFiles[i]);
		}
		else
		{
			//std::cout << "invalid file: " << shaderFolderFiles[i] << std::endl;
		}
	}

	// texture files
	std::vector<std::string> textureFolderFiles = get_all_files_names_within_folder(projectDirectory + "data/textures");
	std::vector<std::string> textureFilePaths;
	for(unsigned int i = 0; i < textureFolderFiles.size(); i++){
		if(textureFolderFiles[i].substr(textureFolderFiles[i].find_last_of(".") + 1) == "json") {
			textureFilePaths.push_back(projectDirectory + "data/textures/" + textureFolderFiles[i]);
		}
		else
		{
			//std::cout << "invalid file: " << textureFolderFiles[i] << std::endl;
		}
	}

	// material files
	std::vector<std::string> materialFolderFiles = get_all_files_names_within_folder(projectDirectory + "data/materials");
	std::vector<std::string> materialFilePaths;
	for(unsigned int i = 0; i < materialFolderFiles.size(); i++){
		if(materialFolderFiles[i].substr(materialFolderFiles[i].find_last_of(".") + 1) == "json") {
			materialFilePaths.push_back(projectDirectory + "data/materials/" + materialFolderFiles[i]);
		}
		else
		{
			//std::cout << "invalid file: " << materialFolderFiles[i] << std::endl;
		}
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
			//std::cout << "invalid file: " << meshFolderFiles[i] << std::endl;
		}
	}

	// gmesh files
	std::vector<std::string> gmeshFolderFiles = get_all_files_names_within_folder(projectDirectory + "data/gmeshes");
	std::vector<std::string> gmeshFilePaths;
	for(unsigned int i = 0; i < gmeshFolderFiles.size(); i++){
		if(gmeshFolderFiles[i].substr(gmeshFolderFiles[i].find_last_of(".") + 1) == "json") {
			gmeshFilePaths.push_back(projectDirectory + "data/gmeshes/" + gmeshFolderFiles[i]);
		}
		else
		{
			//std::cout << "invalid file: " << gmeshFolderFiles[i] << std::endl;
		}
	}

	// font files
	std::vector<std::string> fontFolderFiles = get_all_files_names_within_folder(projectDirectory + "data/fonts");
	std::vector<std::string> fontFilePaths;
	for(unsigned int i = 0; i < fontFolderFiles.size(); i++){
		if(fontFolderFiles[i].substr(fontFolderFiles[i].find_last_of(".") + 1) == "json") {
			fontFilePaths.push_back(projectDirectory + "data/fonts/" + fontFolderFiles[i]);
		}
		else
		{
			//std::cout << "invalid file: " << fontFolderFiles[i] << std::endl;
		}
	}

	// write shaders out to bundle
	for(size_t i = 0; i < shaderFilePaths.size(); i++){
		// open json file and load to json object
		std::ifstream in(shaderFilePaths[i], std::ios::in | std::ios::binary);
		std::ostringstream contents;
		contents << in.rdbuf();
		in.close();

		json::JSON jsonShader = JSON::Load(contents.str());

		Shader shader;

		std::string filePath = shaderFilePaths[i].substr(0, shaderFilePaths[i].find_last_of(".")) + ".shader";

		std::vector<char> data;	
		int assetType = -1;
		if(AssetLoader::load(filePath, shader)){
			shader.assetId = Guid(jsonShader["id"].ToString());
			assetType = AssetType<Shader>::type;
			data = shader.serialize();

			std::cout << "vertex shader length: " << shader.vertexShader.length() << " geometry shader length: " << shader.geometryShader.length() << " fragment shader length: " << shader.fragmentShader.length() << std::endl;
		}	
		else{
			std::cout << "Failed to open file " << filePath << " for parsing" << std::endl;
			return 0;
		}	

		std::string temp = filePath.substr(filePath.find_last_of("/") + 1);
		std::string outFilePath = "assets\\" + temp.substr(0, temp.find_last_of(".")) + ".data";
		std::fstream outFile(outFilePath, std::ios::out | std::ios::binary);

		if(outFile.is_open()){
			char classification = 'a';
			int type = 0;
			size_t size = data.size();

			AssetHeader header = {};
			outFile.write((char*)&header, sizeof(AssetHeader));			

			outFile.write(&classification, 1);
			outFile.write((char*)&assetType, sizeof(int));
			outFile.write((char*)&size, sizeof(size_t));
			outFile.write(&data[0], data.size());

			outFile.close();
		}
		else{
			return 0;
		}
	}

	// write texture out to bundle
	for(size_t i = 0; i < textureFilePaths.size(); i++){
		// open json file and load to json object
		std::ifstream in(textureFilePaths[i], std::ios::in | std::ios::binary);
		std::ostringstream contents;
		contents << in.rdbuf();
		in.close();

		json::JSON jsonTexture = JSON::Load(contents.str());

		Texture2D texture;

		std::string filePath = textureFilePaths[i].substr(0, textureFilePaths[i].find_last_of(".")) + ".png";

		std::cout << "texture: " << filePath << std::endl;

		std::vector<char> data;	
		int assetType = -1;
		if(AssetLoader::load(filePath, texture)){
			texture.assetId = Guid(jsonTexture["id"].ToString());
			assetType = AssetType<Texture2D>::type;
			data = texture.serialize();

			std::cout << "texture raw data size: " << texture.getRawTextureData().size() << std::endl;
		}	
		else{
			std::cout << "Failed to open file " << filePath << " for parsing" << std::endl;
			return 0;
		}	

		std::string temp = filePath.substr(filePath.find_last_of("/") + 1);
		std::string outFilePath = "assets\\" + temp.substr(0, temp.find_last_of(".")) + ".data";
		std::fstream outFile(outFilePath, std::ios::out | std::ios::binary);

		if(outFile.is_open()){
			char classification = 'a';
			int type = 1;
			size_t size = data.size();

			AssetHeader header = {};
			outFile.write((char*)&header, sizeof(AssetHeader));	

			outFile.write(&classification, 1);
			outFile.write((char*)&assetType, sizeof(int));
			outFile.write((char*)&size, sizeof(size_t));
			outFile.write(&data[0], data.size());

			outFile.close();
		}
		else{
			return 0;
		}
	}

	// write materials out to bundle
	for(size_t i = 0; i < materialFilePaths.size(); i++){
		// open json file and load to json object
		std::ifstream in(materialFilePaths[i], std::ios::in | std::ios::binary);
		std::ostringstream contents;
		contents << in.rdbuf();
		in.close();

		json::JSON jsonMaterial = JSON::Load(contents.str());

		Material material;

		std::string filePath = materialFilePaths[i].substr(0, materialFilePaths[i].find_last_of(".")) + ".material";

		std::cout << "material: " << filePath << " id: " << jsonMaterial["id"].ToString() << std::endl;

		std::vector<char> data;	
		int assetType = -1;
		if(AssetLoader::load(filePath, material)){
			material.assetId = Guid(jsonMaterial["id"].ToString());
			assetType = AssetType<Material>::type;
			data = material.serialize();
		}
		else{
			std::cout << "Failed to open file " << filePath << " for parsing" << std::endl;
			return 0;
		}
		
		std::string temp = filePath.substr(filePath.find_last_of("/") + 1);
		std::string outFilePath = "assets\\" + temp.substr(0, temp.find_last_of(".")) + ".data";
		std::fstream outFile(outFilePath, std::ios::out | std::ios::binary);

		if(outFile.is_open()){
			char classification = 'a';
			int type = 4;
			size_t size = data.size();

			AssetHeader header = {};
			outFile.write((char*)&header, sizeof(AssetHeader));	

			outFile.write(&classification, 1);
			outFile.write((char*)&assetType, sizeof(int));
			outFile.write((char*)&size, sizeof(size_t));
			outFile.write(&data[0], data.size());

			outFile.close();
		}
		else{
			return 0;
		}
	}

	// write meshes out to bundle
	for(size_t i = 0 ; i < meshFilePaths.size(); i++){
		// open json file and load to json object
		std::ifstream in(meshFilePaths[i], std::ios::in | std::ios::binary);
		std::ostringstream contents;
		contents << in.rdbuf();
		in.close();

		json::JSON jsonMesh = JSON::Load(contents.str());

		Mesh mesh;

		std::string filePath = meshFilePaths[i].substr(0, meshFilePaths[i].find_last_of(".")) + ".obj";

		std::cout << "mesh filepath: " << filePath << std::endl;

		std::vector<char> data;
		int assetType = -1;
		if(AssetLoader::load(filePath, mesh)){
			mesh.assetId = Guid(jsonMesh["id"].ToString());
			assetType = AssetType<Mesh>::type;
			data = mesh.serialize();
		}
		else{
			std::cout << "Failed to open file " << filePath << " for parsing" << std::endl;
			return 0;
		}

		std::string temp = filePath.substr(filePath.find_last_of("/") + 1);
		std::string outFilePath = "assets\\" + temp.substr(0, temp.find_last_of(".")) + ".data";
		std::fstream outFile(outFilePath, std::ios::out | std::ios::binary);

		std::cout << "out file path: " << outFilePath << std::endl;

		if(outFile.is_open()){
			char classification = 'a';
			int type = 5;
			size_t size = data.size();

			AssetHeader header = {};
			outFile.write((char*)&header, sizeof(AssetHeader));	

			outFile.write(&classification, 1);
			outFile.write((char*)&assetType, sizeof(int));
			outFile.write((char*)&size, sizeof(size_t));
			outFile.write(&data[0], data.size());

			outFile.close();
		}
		else{
			return 0;
		}
	}

	std::cout << "Asset bundle successfully created" << std::endl;

	while(true)
	{

	}

	return 1;




















	// for(unsigned int i = 0; i < materialFilePaths.size(); i++){

	// 	// open json file and load to json object
	// 	std::ifstream in(materialFilePaths[i], std::ios::in | std::ios::binary);
	// 	std::ostringstream contents;
	// 	contents << in.rdbuf();
	// 	in.close();

	// 	std::string jsonString = contents.str();
	// 	json::JSON jsonMaterial = JSON::Load(jsonString);

	// 	MaterialData data;
	// 	data.assetId = Guid(jsonMaterial["id"].ToString());
	// 	data.shininess = (float)jsonMaterial["shininess"].ToFloat();
	// 	data.ambient.x = (float)jsonMaterial["ambient"][0].ToFloat();
	// 	data.ambient.y = (float)jsonMaterial["ambient"][1].ToFloat();
	// 	data.ambient.z = (float)jsonMaterial["ambient"][2].ToFloat();
	// 	data.diffuse.x = (float)jsonMaterial["diffuse"][0].ToFloat();
	// 	data.diffuse.y = (float)jsonMaterial["diffuse"][1].ToFloat();
	// 	data.diffuse.z = (float)jsonMaterial["diffuse"][2].ToFloat();
	// 	data.specular.x = (float)jsonMaterial["specular"][0].ToFloat();
	// 	data.specular.y = (float)jsonMaterial["specular"][1].ToFloat();
	// 	data.specular.z = (float)jsonMaterial["specular"][2].ToFloat();
	// 	data.shaderId  = Guid(jsonMaterial["shader"].ToString());
	// 	data.textureId = Guid(jsonMaterial["mainTexture"].ToString());
	// 	data.normalMapId = Guid(jsonMaterial["normalMap"].ToString());
	// 	data.specularMapId = Guid(jsonMaterial["specularMap"].ToString());

	// 	std::string outputPath = materialFilePaths[i].substr(0, materialFilePaths[i].find_last_of(".")) + ".mat";

	// 	std::cout << "outputPath: " << outputPath << " material id: " << data.assetId.toString() << " shader id: " << data.shaderId.toString() << " main texture id: " << data.textureId.toString() << std::endl;

	// 	// serialize material
	// 	FILE* file = fopen(outputPath.c_str(), "wb");
	// 	if (file){
	// 		size_t test = fwrite(&data, sizeof(MaterialData), 1, file);
	// 		std::cout << "number of bytes written to file: " << test << std::endl;
	// 	}
	// 	else{
	// 		std::cout << "Failed to open file " << outputPath << " for writing." << std::endl;
	// 		return 0;
	// 	}

	// 	if(file){
	// 		fclose(file);
	// 	}
	// }

	// for(unsigned int i = 0 ; i < meshFilePaths.size(); i++){
	// 	// open json file and load to json object
	// 	std::ifstream in(meshFilePaths[i], std::ios::in | std::ios::binary);
	// 	std::ostringstream contents;
	// 	contents << in.rdbuf();
	// 	in.close();

	// 	std::string jsonString = contents.str();
	// 	json::JSON jsonMesh = JSON::Load(jsonString);

	// 	Mesh mesh;

	// 	std::string filePath = meshFilePaths[i].substr(0, meshFilePaths[i].find_last_of(".")) + ".txt";

	// 	std::cout << "mesh filepath: " << filePath << std::endl;

	// 	if(MeshLoader::load(filePath, mesh)){
			
	// 		// create mesh header
	// 		MeshHeader header = {};
	// 		header.meshId = Guid(jsonMesh["id"].ToString());
	// 		header.verticesSize = (unsigned int)mesh.vertices.size();
	// 		header.normalsSize = (unsigned int)mesh.normals.size();
	// 		header.texCoordsSize = (unsigned int)mesh.texCoords.size();

	// 		std::cout << "vertices size: " << mesh.vertices.size() << " normals size: " << mesh.normals.size() << " texCoords size: " << mesh.texCoords.size() << std::endl;

	// 		std::string outputPath = meshFilePaths[i].substr(0, meshFilePaths[i].find_last_of(".")) + ".mesh";

	// 		// serialize scene header and mesh data
	// 		FILE* file = fopen(outputPath.c_str(), "wb");
	// 		if (file){
	// 			size_t test = fwrite(&header, sizeof(MeshHeader), 1, file);
	// 			test += fwrite(&(mesh.vertices[0]), mesh.vertices.size()*sizeof(float), 1, file);
	// 			test += fwrite(&(mesh.normals[0]), mesh.normals.size()*sizeof(float), 1, file);
	// 			test += fwrite(&(mesh.texCoords[0]), mesh.texCoords.size()*sizeof(float), 1, file);
	// 			std::cout << "number of bytes written to file: " << test << std::endl;
	// 		}
	// 		else{
	// 			std::cout << "Failed to open file " << outputPath << " for writing." << std::endl;
	// 			return 0;
	// 		}

	// 		if(file){
	// 			fclose(file);
	// 		}
	// 	}
	// 	else{
	// 		std::cout << "Failed to open file " << filePath << " for parsing" << std::endl;
	// 		return 0;
	// 	}
	// }

	// for(unsigned int i = 0 ; i < gmeshFilePaths.size(); i++){
	// // open json file and load to json object
	// 	std::ifstream in(gmeshFilePaths[i], std::ios::in | std::ios::binary);
	// 	std::ostringstream contents;
	// 	contents << in.rdbuf();
	// 	in.close();

	// 	std::string jsonString = contents.str();
	// 	json::JSON jsonGMesh = JSON::Load(jsonString);

	// 	GMesh gmesh;

	// 	std::string filePath = gmeshFilePaths[i].substr(0, gmeshFilePaths[i].find_last_of(".")) + ".msh";

	// 	if(MeshLoader::load(filePath, gmesh)){
			
	// 		// create gmesh header
	// 		GMeshHeader header = {};
	// 		header.gmeshId = Guid(jsonGMesh["id"].ToString());
	// 		header.dim = gmesh.dim;
	// 		header.ng = gmesh.ng;
	// 	    header.n = gmesh.n;
	// 	    header.nte = gmesh.nte;
	// 	    header.ne = gmesh.ne;
	// 	    header.ne_b = gmesh.ne_b;
	// 	    header.npe = gmesh.npe;
	// 	    header.npe_b = gmesh.npe_b;
	// 	    header.type = gmesh.type;
	// 	    header.type_b = gmesh.type_b;
	// 		header.verticesSize = (unsigned int)gmesh.vertices.size();
	// 		header.connectSize = (unsigned int)gmesh.connect.size();
	// 		header.bconnectSize = (unsigned int)gmesh.bconnect.size();
	// 		header.groupsSize = (unsigned int)gmesh.groups.size();

	// 		std::cout << "vertices size: " << gmesh.vertices.size() << " connect size: " << gmesh.connect.size() << " bconnect size: " << gmesh.bconnect.size() << " groups size: " << gmesh.groups.size() << std::endl;

	// 		std::string outputPath = gmeshFilePaths[i].substr(0, gmeshFilePaths[i].find_last_of(".")) + ".gmesh";

	// 		// serialize scene header and mesh data
	// 		FILE* file = fopen(outputPath.c_str(), "wb");
	// 		if (file){
	// 			size_t test = fwrite(&header, sizeof(GMeshHeader), 1, file);
	// 			test += fwrite(&gmesh.vertices[0], gmesh.vertices.size()*sizeof(float), 1, file);
	// 			test += fwrite(&gmesh.connect[0], gmesh.connect.size()*sizeof(float), 1, file);
	// 			test += fwrite(&gmesh.bconnect[0], gmesh.bconnect.size()*sizeof(float), 1, file);
	// 			test += fwrite(&gmesh.groups[0], gmesh.groups.size()*sizeof(float), 1, file);
	// 			std::cout << "number of bytes written to file: " << test << std::endl;
	// 		}
	// 		else{
	// 			std::cout << "Failed to open file " << outputPath << " for writing." << std::endl;
	// 			return 0;
	// 		}

	// 		if(file){
	// 			fclose(file);
	// 		}
	// 	}
	// 	else{
	// 		std::cout << "Failed to open file " << filePath << " for parsing" << std::endl;
	// 		return 0;
	// 	}
	// }

	// while(true)
	// {

	// }

	// return 1;
}



// int serializeMaterials(std::vector<std::string> materialFilePaths)
// {
// 	for(unsigned int i = 0; i < materialFilePaths.size(); i++){

// 		// open json file and load to json object
// 		std::ifstream in(materialFilePaths[i], std::ios::in | std::ios::binary);
// 		std::ostringstream contents;
// 		contents << in.rdbuf();
// 		in.close();

// 		std::string jsonString = contents.str();
// 		json::JSON jsonMaterial = JSON::Load(jsonString);

// 		MaterialData data;
// 		data.assetId = Guid(jsonMaterial["id"].ToString());
// 		data.shininess = (float)jsonMaterial["shininess"].ToFloat();
// 		data.ambient.x = (float)jsonMaterial["ambient"][0].ToFloat();
// 		data.ambient.y = (float)jsonMaterial["ambient"][1].ToFloat();
// 		data.ambient.z = (float)jsonMaterial["ambient"][2].ToFloat();
// 		data.diffuse.x = (float)jsonMaterial["diffuse"][0].ToFloat();
// 		data.diffuse.y = (float)jsonMaterial["diffuse"][1].ToFloat();
// 		data.diffuse.z = (float)jsonMaterial["diffuse"][2].ToFloat();
// 		data.specular.x = (float)jsonMaterial["specular"][0].ToFloat();
// 		data.specular.y = (float)jsonMaterial["specular"][1].ToFloat();
// 		data.specular.z = (float)jsonMaterial["specular"][2].ToFloat();
// 		data.shaderId  = Guid(jsonMaterial["shader"].ToString());
// 		data.textureId = Guid(jsonMaterial["mainTexture"].ToString());
// 		data.normalMapId = Guid(jsonMaterial["normalMap"].ToString());
// 		data.specularMapId = Guid(jsonMaterial["specularMap"].ToString());

// 		std::string outputPath = materialFilePaths[i].substr(0, materialFilePaths[i].find_last_of(".")) + ".mat";

// 		std::cout << "outputPath: " << outputPath << " material id: " << data.assetId.toString() << " shader id: " << data.shaderId.toString() << " main texture id: " << data.textureId.toString() << std::endl;

// 		// serialize material
// 		FILE* file = fopen(outputPath.c_str(), "wb");
// 		if (file){
// 			size_t test = fwrite(&data, sizeof(MaterialData), 1, file);
// 			std::cout << "number of bytes written to file: " << test << std::endl;
// 		}
// 		else{
// 			std::cout << "Failed to open file " << outputPath << " for writing." << std::endl;
// 			return 0;
// 		}

// 		if(file){
// 			fclose(file);
// 		}
// 	}

// 	return 1;
// }

// int serializeMeshes(std::vector<std::string> meshFilePaths)
// {
// 	for(unsigned int i = 0 ; i < meshFilePaths.size(); i++){
// 		// open json file and load to json object
// 		std::ifstream in(meshFilePaths[i], std::ios::in | std::ios::binary);
// 		std::ostringstream contents;
// 		contents << in.rdbuf();
// 		in.close();

// 		std::string jsonString = contents.str();
// 		json::JSON jsonMesh = JSON::Load(jsonString);

// 		Mesh mesh;

// 		std::string filePath = meshFilePaths[i].substr(0, meshFilePaths[i].find_last_of(".")) + ".txt";

// 		std::cout << "mesh filepath: " << filePath << std::endl;

// 		if(MeshLoader::load(filePath, mesh)){
			
// 			// create mesh header
// 			MeshHeader header = {};
// 			header.meshId = Guid(jsonMesh["id"].ToString());
// 			header.verticesSize = (unsigned int)mesh.vertices.size();
// 			header.normalsSize = (unsigned int)mesh.normals.size();
// 			header.texCoordsSize = (unsigned int)mesh.texCoords.size();

// 			std::cout << "vertices size: " << mesh.vertices.size() << " normals size: " << mesh.normals.size() << " texCoords size: " << mesh.texCoords.size() << std::endl;

// 			std::string outputPath = meshFilePaths[i].substr(0, meshFilePaths[i].find_last_of(".")) + ".mesh";

// 			// serialize scene header and mesh data
// 			FILE* file = fopen(outputPath.c_str(), "wb");
// 			if (file){
// 				size_t test = fwrite(&header, sizeof(MeshHeader), 1, file);
// 				test += fwrite(&(mesh.vertices[0]), mesh.vertices.size()*sizeof(float), 1, file);
// 				test += fwrite(&(mesh.normals[0]), mesh.normals.size()*sizeof(float), 1, file);
// 				test += fwrite(&(mesh.texCoords[0]), mesh.texCoords.size()*sizeof(float), 1, file);
// 				std::cout << "number of bytes written to file: " << test << std::endl;
// 			}
// 			else{
// 				std::cout << "Failed to open file " << outputPath << " for writing." << std::endl;
// 				return 0;
// 			}

// 			if(file){
// 				fclose(file);
// 			}
// 		}
// 		else{
// 			std::cout << "Failed to open file " << filePath << " for parsing" << std::endl;
// 			return 0;
// 		}
// 	}

// 	return 1;
// }

// int serializeGMeshes(std::vector<std::string> gmeshFilePaths)
// {
// 	for(unsigned int i = 0 ; i < gmeshFilePaths.size(); i++){
// 	// open json file and load to json object
// 		std::ifstream in(gmeshFilePaths[i], std::ios::in | std::ios::binary);
// 		std::ostringstream contents;
// 		contents << in.rdbuf();
// 		in.close();

// 		std::string jsonString = contents.str();
// 		json::JSON jsonGMesh = JSON::Load(jsonString);

// 		GMesh gmesh;

// 		std::string filePath = gmeshFilePaths[i].substr(0, gmeshFilePaths[i].find_last_of(".")) + ".msh";

// 		if(MeshLoader::load(filePath, gmesh)){
			
// 			// create gmesh header
// 			GMeshHeader header = {};
// 			header.gmeshId = Guid(jsonGMesh["id"].ToString());
// 			header.dim = gmesh.dim;
// 			header.ng = gmesh.ng;
// 		    header.n = gmesh.n;
// 		    header.nte = gmesh.nte;
// 		    header.ne = gmesh.ne;
// 		    header.ne_b = gmesh.ne_b;
// 		    header.npe = gmesh.npe;
// 		    header.npe_b = gmesh.npe_b;
// 		    header.type = gmesh.type;
// 		    header.type_b = gmesh.type_b;
// 			header.verticesSize = (unsigned int)gmesh.vertices.size();
// 			header.connectSize = (unsigned int)gmesh.connect.size();
// 			header.bconnectSize = (unsigned int)gmesh.bconnect.size();
// 			header.groupsSize = (unsigned int)gmesh.groups.size();

// 			std::cout << "vertices size: " << gmesh.vertices.size() << " connect size: " << gmesh.connect.size() << " bconnect size: " << gmesh.bconnect.size() << " groups size: " << gmesh.groups.size() << std::endl;

// 			std::string outputPath = gmeshFilePaths[i].substr(0, gmeshFilePaths[i].find_last_of(".")) + ".gmesh";

// 			// serialize scene header and mesh data
// 			FILE* file = fopen(outputPath.c_str(), "wb");
// 			if (file){
// 				size_t test = fwrite(&header, sizeof(GMeshHeader), 1, file);
// 				test += fwrite(&gmesh.vertices[0], gmesh.vertices.size()*sizeof(float), 1, file);
// 				test += fwrite(&gmesh.connect[0], gmesh.connect.size()*sizeof(float), 1, file);
// 				test += fwrite(&gmesh.bconnect[0], gmesh.bconnect.size()*sizeof(float), 1, file);
// 				test += fwrite(&gmesh.groups[0], gmesh.groups.size()*sizeof(float), 1, file);
// 				std::cout << "number of bytes written to file: " << test << std::endl;
// 			}
// 			else{
// 				std::cout << "Failed to open file " << outputPath << " for writing." << std::endl;
// 				return 0;
// 			}

// 			if(file){
// 				fclose(file);
// 			}
// 		}
// 		else{
// 			std::cout << "Failed to open file " << filePath << " for parsing" << std::endl;
// 			return 0;
// 		}
// 	}

// 	return 1;
// }