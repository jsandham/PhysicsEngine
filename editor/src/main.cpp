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
#include <components/Boids.h>

#include <systems/PhysicsSystem.h>
#include <systems/RenderSystem.h>

#include <json/json.hpp>

#include "../../sample_project/Demo/Demo/include/LogicSystem.h"
#include "../../sample_project/Demo/Demo/include/PlayerSystem.h"

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
	std::string projectDirectory = "../../sample_project/Demo/Demo/";

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
		if(sceneFolderFiles[i].substr(sceneFolderFiles[i].find_last_of(".") + 1) == "json") {
			sceneFilePaths.push_back(projectDirectory + "data/scenes/" + sceneFolderFiles[i]);
		}
		else
		{
			std::cout << "invalid file: " << sceneFolderFiles[i] << std::endl;
		}
	}

	// loop through all scenes and serialize them
	for(size_t i = 0; i < sceneFilePaths.size(); i++){
		std::string outputPath = sceneFilePaths[i].substr(sceneFilePaths[i].find_last_of("\\/") + 1);
		outputPath = outputPath.substr(0, outputPath.find_last_of(".")) + ".scene";
		std::cout << "output path: " << outputPath << std::endl;

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
		json::JSON directionalLights;
		json::JSON spotLights;
		json::JSON pointLights;
		json::JSON boxColliders;
		json::JSON sphereColliders;
		json::JSON capsuleColliders;
		json::JSON boids;
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
			else if(type == "Boids"){
				boids[it->first] = it->second;
			}
			else if(type == "PhysicsSystem"){
				//std::cout << it->first << " is a PhysicsSystem" << std::endl;
				systems[it->first] = it->second;
			}
			else if(type == "RenderSystem"){
				//std::cout << it->first << " is a RenderSystem" << std::endl;
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
			else if(type == "BoidsSystem"){
				//std::cout << it->first << " is a LogicSystem" << std::endl;
				systems[it->first] = it->second;
			}
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
		unsigned int numberOfDirectionalLights = std::max(0, directionalLights.size());
		unsigned int numberOfSpotLights = std::max(0, spotLights.size());
		unsigned int numberOfPointLights = std::max(0, pointLights.size());
		unsigned int numberOfBoxColliders = std::max(0, boxColliders.size());
		unsigned int numberOfSphereColliders = std::max(0, sphereColliders.size());
		unsigned int numberOfCapsuleColliders = std::max(0, capsuleColliders.size());
		unsigned int numberOfBoids = std::max(0, boids.size());
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
		std::cout << "number of boids found: " << numberOfBoids << std::endl;
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
		header.numberOfBoids = numberOfBoids;
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
		header.sizeOfBoids = sizeof(Boids);

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
			EntityHeader data;

			data.entityId = Guid(it->first);

			int type = 0;
			char classification = 'e';

			size_t totalSize = sizeof(int);
			totalSize += sizeof(char);
			totalSize += sizeof(EntityHeader);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&classification, sizeof(char), 1, file);
			fwrite(&data, sizeof(EntityHeader), 1, file);
		}

		// serialize transforms
		objects = transforms.ObjectRange();
		for(it = objects.begin(); it != objects.end(); it++){
			TransformHeader data;

			data.componentId = Guid(it->first);
			data.entityId = Guid(it->second["entity"].ToString());

			data.position.x = (float)it->second["position"][0].ToFloat();
			data.position.y = (float)it->second["position"][1].ToFloat();
			data.position.z = (float)it->second["position"][2].ToFloat();

			data.rotation.x = (float)it->second["rotation"][0].ToFloat();
			data.rotation.y = (float)it->second["rotation"][1].ToFloat();
			data.rotation.z = (float)it->second["rotation"][2].ToFloat();
			data.rotation.w = (float)it->second["rotation"][3].ToFloat();

			data.scale.x = (float)it->second["scale"][0].ToFloat();
			data.scale.y = (float)it->second["scale"][1].ToFloat();
			data.scale.z = (float)it->second["scale"][2].ToFloat();

			int type = 0;
			char classification = 'c';

			size_t totalSize = sizeof(int);
			totalSize += sizeof(char);
			totalSize += sizeof(TransformHeader);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&classification, sizeof(char), 1, file);

			fwrite(&data, sizeof(TransformHeader), 1, file);
		}

		// serialize rigidbodies
		objects = rigidbodies.ObjectRange();
		for(it = objects.begin(); it != objects.end(); it++){
			RigidbodyHeader data;

			data.componentId = Guid(it->first);
			data.entityId = Guid(it->second["entity"].ToString());

			data.useGravity = (bool)it->second["useGravity"].ToBool();
			data.mass = (float)it->second["mass"].ToFloat();
			data.drag = (float)it->second["drag"].ToFloat();
			data.angularDrag = (float)it->second["angularDrag"].ToFloat();

			data.velocity.x = (float)it->second["velocity"][0].ToFloat();
			data.velocity.y = (float)it->second["velocity"][1].ToFloat();
			data.velocity.z = (float)it->second["velocity"][2].ToFloat();

			data.angularVelocity.x = (float)it->second["angularVelocity"][0].ToFloat();
			data.angularVelocity.y = (float)it->second["angularVelocity"][1].ToFloat();
			data.angularVelocity.z = (float)it->second["angularVelocity"][2].ToFloat();

			data.centreOfMass.x = (float)it->second["centreOfMass"][0].ToFloat();
			data.centreOfMass.y = (float)it->second["centreOfMass"][1].ToFloat();
			data.centreOfMass.z = (float)it->second["centreOfMass"][2].ToFloat();

			data.inertiaTensor = glm::mat3(1.0f);
			data.halfVelocity = glm::vec3(0.0f, 0.0f,0.0f);

			int type = 1;
			char classification = 'c';

			size_t totalSize = sizeof(int);
			totalSize += sizeof(char);
			totalSize += sizeof(RigidbodyHeader);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&classification, sizeof(char), 1, file);

			fwrite(&data, sizeof(RigidbodyHeader), 1, file);
		}

		// serialize cameras 
		objects = cameras.ObjectRange();
		for(it = objects.begin(); it != objects.end(); it++){
			CameraHeader data;

			data.componentId = Guid(it->first);
			data.entityId = Guid(it->second["entity"].ToString());

			data.position.x = (float)it->second["position"][0].ToFloat();
			data.position.y = (float)it->second["position"][1].ToFloat();
			data.position.z = (float)it->second["position"][2].ToFloat();

			data.backgroundColor.x = (float)it->second["backgroundColor"][0].ToFloat();
			data.backgroundColor.y = (float)it->second["backgroundColor"][1].ToFloat();
			data.backgroundColor.z = (float)it->second["backgroundColor"][2].ToFloat();
			data.backgroundColor.w = (float)it->second["backgroundColor"][3].ToFloat();

			int type = 2;
			char classification = 'c';

			size_t totalSize = sizeof(int);
			totalSize += sizeof(char);
			totalSize += sizeof(CameraHeader);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&classification, sizeof(char), 1, file);

			fwrite(&data, sizeof(CameraHeader), 1, file);
		}

		// serialize mesh renderers
		objects = meshRenderers.ObjectRange();
		for(it = objects.begin(); it != objects.end(); it++){
			MeshRendererHeader data;

			data.componentId = Guid(it->first);
			data.entityId = Guid(it->second["entity"].ToString());

			data.meshId = Guid(it->second["mesh"].ToString());
			data.materialId = Guid(it->second["material"].ToString());

			int type = 3;
			char classification = 'c';

			size_t totalSize = sizeof(int);
			totalSize += sizeof(char);
			totalSize += sizeof(MeshRendererHeader);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&classification, sizeof(char), 1, file);

			fwrite(&data, sizeof(MeshRendererHeader), 1, file);
		}

		std::cout << "size of mesh renderer: " << sizeof(MeshRendererHeader) << std::endl;

		// serialize line renderers
		objects = lineRenderers.ObjectRange();
		for(it = objects.begin(); it != objects.end(); it++){
			LineRendererHeader data;

			data.componentId = Guid(it->first);
			data.entityId = Guid(it->second["entity"].ToString());
			data.materialId = Guid(it->second["material"].ToString());

			data.start.x = (float)it->second["start"][0].ToFloat();
			data.start.y = (float)it->second["start"][1].ToFloat();
			data.start.z = (float)it->second["start"][2].ToFloat();

			data.end.x = (float)it->second["end"][0].ToFloat();
			data.end.y = (float)it->second["end"][1].ToFloat();
			data.end.z = (float)it->second["end"][2].ToFloat();

			// data.color.x = (float)it->second["color"][0].ToFloat();
			// data.color.y = (float)it->second["color"][1].ToFloat();
			// data.color.z = (float)it->second["color"][2].ToFloat();
			// data.color.w = (float)it->second["color"][3].ToFloat();

			//std::cout << "line renderer entity id: " << data.entityId.toString() << "line renderer component id: " << data.componentId.toString() << std::endl;

			int type = 4;
			char classification = 'c';

			size_t totalSize = sizeof(int);
			totalSize += sizeof(char);
			totalSize += sizeof(LineRendererHeader);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&classification, sizeof(char), 1, file);

			fwrite(&data, sizeof(LineRendererHeader), 1, file);
		}

		std::cout << "size of line renderer: " << sizeof(LineRenderer) << std::endl;

		// serialize directional lights
		objects = directionalLights.ObjectRange();
		for(it = objects.begin(); it != objects.end(); it++){
			DirectionalLightHeader data;

			data.componentId = Guid(it->first);
			data.entityId = Guid(it->second["entity"].ToString());

			data.direction.x = (float)it->second["direction"][0].ToFloat();
			data.direction.y = (float)it->second["direction"][1].ToFloat();
			data.direction.z = (float)it->second["direction"][2].ToFloat();

			data.ambient.x = (float)it->second["ambient"][0].ToFloat();
			data.ambient.y = (float)it->second["ambient"][1].ToFloat();
			data.ambient.z = (float)it->second["ambient"][2].ToFloat();

			data.diffuse.x = (float)it->second["diffuse"][0].ToFloat();
			data.diffuse.y = (float)it->second["diffuse"][1].ToFloat();
			data.diffuse.z = (float)it->second["diffuse"][2].ToFloat();

			data.specular.x = (float)it->second["specular"][0].ToFloat();
			data.specular.y = (float)it->second["specular"][1].ToFloat();
			data.specular.z = (float)it->second["specular"][2].ToFloat();

			int type = 5;
			char classification = 'c';

			size_t totalSize = sizeof(int);
			totalSize += sizeof(char);
			totalSize += sizeof(DirectionalLightHeader);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&classification, sizeof(char), 1, file);

			fwrite(&data, sizeof(DirectionalLightHeader), 1, file);
		}

		// serialize spot lights
		objects = spotLights.ObjectRange();
		for(it = objects.begin(); it != objects.end(); it++){
			SpotLightHeader data;

			data.componentId = Guid(it->first);
			data.entityId = Guid(it->second["entity"].ToString());

			data.constant = (float)it->second["constant"].ToFloat();
			data.linear = (float)it->second["linear"].ToFloat();
			data.quadratic = (float)it->second["quadratic"].ToFloat();
			data.cutOff = (float)it->second["cutOff"].ToFloat();
			data.outerCutOff = (float)it->second["outerCutOff"].ToFloat();

			data.position.x = (float)it->second["position"][0].ToFloat();
			data.position.y = (float)it->second["position"][1].ToFloat();
			data.position.z = (float)it->second["position"][2].ToFloat();

			data.direction.x = (float)it->second["direction"][0].ToFloat();
			data.direction.y = (float)it->second["direction"][1].ToFloat();
			data.direction.z = (float)it->second["direction"][2].ToFloat();

			data.ambient.x = (float)it->second["ambient"][0].ToFloat();
			data.ambient.y = (float)it->second["ambient"][1].ToFloat();
			data.ambient.z = (float)it->second["ambient"][2].ToFloat();

			data.diffuse.x = (float)it->second["diffuse"][0].ToFloat();
			data.diffuse.y = (float)it->second["diffuse"][1].ToFloat();
			data.diffuse.z = (float)it->second["diffuse"][2].ToFloat();

			data.specular.x = (float)it->second["specular"][0].ToFloat();
			data.specular.y = (float)it->second["specular"][1].ToFloat();
			data.specular.z = (float)it->second["specular"][2].ToFloat();

			data.projection = glm::perspective(glm::radians(45.0f), 1.0f * 640 / 480, 0.1f, 100.0f);

			int type = 6;
			char classification = 'c';

			size_t totalSize = sizeof(int);
			totalSize += sizeof(char);
			totalSize += sizeof(SpotLightHeader);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&classification, sizeof(char), 1, file);

			fwrite(&data, sizeof(SpotLightHeader), 1, file);
		}

		// serialize point lights
		objects = pointLights.ObjectRange();
		for(it = objects.begin(); it != objects.end(); it++){
			PointLightHeader data;

			data.componentId = Guid(it->first);
			data.entityId = Guid(it->second["entity"].ToString());

			data.constant = (float)it->second["constant"].ToFloat();
			data.linear = (float)it->second["linear"].ToFloat();
			data.quadratic = (float)it->second["quadratic"].ToFloat();

			data.position.x = (float)it->second["position"][0].ToFloat();
			data.position.y = (float)it->second["position"][1].ToFloat();
			data.position.z = (float)it->second["position"][2].ToFloat();

			data.ambient.x = (float)it->second["ambient"][0].ToFloat();
			data.ambient.y = (float)it->second["ambient"][1].ToFloat();
			data.ambient.z = (float)it->second["ambient"][2].ToFloat();

			data.diffuse.x = (float)it->second["diffuse"][0].ToFloat();
			data.diffuse.y = (float)it->second["diffuse"][1].ToFloat();
			data.diffuse.z = (float)it->second["diffuse"][2].ToFloat();

			data.specular.x = (float)it->second["specular"][0].ToFloat();
			data.specular.y = (float)it->second["specular"][1].ToFloat();
			data.specular.z = (float)it->second["specular"][2].ToFloat();

			data.projection = glm::perspective(glm::radians(45.0f), 1.0f * 640 / 480, 0.1f, 100.0f);

			int type = 7;
			char classification = 'c';

			size_t totalSize = sizeof(int);
			totalSize += sizeof(char);
			totalSize += sizeof(PointLightHeader);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&classification, sizeof(char), 1, file);

			fwrite(&data, sizeof(PointLightHeader), 1, file);
		}

		// serialize box collider
		objects = boxColliders.ObjectRange();
		for(it = objects.begin(); it != objects.end(); it++){
			BoxColliderHeader data;

			data.componentId = Guid(it->first);
			data.entityId = Guid(it->second["entity"].ToString());

			data.bounds.centre.x = (float)it->second["centre"][0].ToFloat();
			data.bounds.centre.y = (float)it->second["centre"][1].ToFloat();
			data.bounds.centre.z = (float)it->second["centre"][2].ToFloat();

			data.bounds.size.x = (float)it->second["size"][0].ToFloat();
			data.bounds.size.y = (float)it->second["size"][1].ToFloat();
			data.bounds.size.z = (float)it->second["size"][2].ToFloat();

			int type = 8;
			char classification = 'c';

			size_t totalSize = sizeof(int);
			totalSize += sizeof(char);
			totalSize += sizeof(BoxColliderHeader);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&classification, sizeof(char), 1, file);

			fwrite(&data, sizeof(BoxColliderHeader), 1, file);
		}

		// serialize sphere collider
		objects = sphereColliders.ObjectRange();
		for(it = objects.begin(); it != objects.end(); it++){
			SphereColliderHeader data;

			data.componentId = Guid(it->first);
			data.entityId = Guid(it->second["entity"].ToString());

			data.sphere.centre.x = (float)it->second["centre"][0].ToFloat();
			data.sphere.centre.y = (float)it->second["centre"][1].ToFloat();
			data.sphere.centre.z = (float)it->second["centre"][2].ToFloat();

			data.sphere.radius = (float)it->second["radius"].ToFloat();

			int type = 9;
			char classification = 'c';

			size_t totalSize = sizeof(int);
			totalSize += sizeof(char);
			totalSize += sizeof(SphereColliderHeader);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&classification, sizeof(char), 1, file);

			fwrite(&data, sizeof(SphereColliderHeader), 1, file);
		}

		// serialize capsule collider
		objects = capsuleColliders.ObjectRange();
		for(it = objects.begin(); it != objects.end(); it++){
			CapsuleColliderHeader data;

			data.componentId = Guid(it->first);
			data.entityId = Guid(it->second["entity"].ToString());

			data.capsule.centre.x = (float)it->second["centre"][0].ToFloat();
			data.capsule.centre.x = (float)it->second["centre"][1].ToFloat();
			data.capsule.centre.x = (float)it->second["centre"][2].ToFloat();

			data.capsule.radius = (float)it->second["radius"].ToFloat();
			data.capsule.height = (float)it->second["height"].ToFloat();

			int type = 10;
			char classification = 'c';

			size_t totalSize = sizeof(int);
			totalSize += sizeof(char);
			totalSize += sizeof(CapsuleColliderHeader);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&classification, sizeof(char), 1, file);

			fwrite(&data, sizeof(CapsuleColliderHeader), 1, file);
		}

		// serialize boids
		objects = boids.ObjectRange();
		for(it = objects.begin(); it != objects.end(); it++){
			BoidsHeader data;

			data.componentId = Guid(it->first);
			data.entityId = Guid(it->second["entity"].ToString());
			data.meshId = Guid(it->second["mesh"].ToString());
			data.materialId = Guid(it->second["material"].ToString());

			data.numBoids = it->second["numBoids"].ToInt();
			data.h = (float)it->second["h"].ToFloat();
			data.bounds.centre.x = (float)it->second["centre"][0].ToFloat();
			data.bounds.centre.y = (float)it->second["centre"][1].ToFloat();
			data.bounds.centre.z = (float)it->second["centre"][2].ToFloat();

			data.bounds.size.x = (float)it->second["size"][0].ToFloat();
			data.bounds.size.y = (float)it->second["size"][1].ToFloat();
			data.bounds.size.z = (float)it->second["size"][2].ToFloat();

			int type = 11;
			char classification = 'c';

			size_t totalSize = sizeof(int);
			totalSize += sizeof(char);
			totalSize += sizeof(BoidsHeader);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&classification, sizeof(char), 1, file);

			fwrite(&data, sizeof(BoidsHeader), 1, file);
		}

		// serialize systems;
		objects = systems.ObjectRange();
		for(it = objects.begin(); it != objects.end(); it++){
			size_t totalSize;
			int type;
			char classification = 's';

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
			else if(it->second["type"].ToString() == "BoidsSystem"){
				type = 4;
			}
			else if(it->second["type"].ToString() == "LogicSystem"){
				type = 20;
			}
			else if(it->second["type"].ToString() == "PlayerSystem"){
				type = 21;
			}

			totalSize = sizeof(int);
			totalSize += sizeof(char);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&classification, sizeof(char), 1, file);
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
		std::cout << "numberOfBoids: " << sceneHeader.numberOfBoids << std::endl;

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
		std::cout << "sizeOfBoids: " << sceneHeader.sizeOfBoids << std::endl;

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
			std::cout << "invalid file: " << shaderFolderFiles[i] << std::endl;
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
			std::cout << "invalid file: " << textureFolderFiles[i] << std::endl;
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
			std::cout << "invalid file: " << materialFolderFiles[i] << std::endl;
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
			std::cout << "invalid file: " << meshFolderFiles[i] << std::endl;
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
			std::cout << "invalid file: " << gmeshFolderFiles[i] << std::endl;
		}
	}

	AssetBundleHeader bundle;
	bundle.numberOfShaders = (unsigned int)shaderFilePaths.size();
	bundle.numberOfTextures = (unsigned int)textureFilePaths.size();
	bundle.numberOfMaterials = (unsigned int)materialFilePaths.size();
	bundle.numberOfMeshes = (unsigned int)meshFilePaths.size();
	bundle.numberOfGMeshes = (unsigned int)gmeshFilePaths.size();

	std::string bundleFileName = "bundle.assets";

	FILE* file = fopen(bundleFileName.c_str(), "wb");
	if(!file){
		std::cout << "Failed to open file " << bundleFileName << " for writing" << std::endl;
		return 0;
	}

	fwrite(&bundle, sizeof(AssetBundleHeader), 1, file);

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

		if(AssetLoader::load(filePath, shader)){
			ShaderHeader header;
			header.shaderId = Guid(jsonShader["id"].ToString());
			header.vertexShaderSize = shader.vertexShader.length();
			header.geometryShaderSize = shader.geometryShader.length();
			header.fragmentShaderSize = shader.fragmentShader.length();

			int type = 0;

			size_t totalSize = sizeof(int);
			totalSize += sizeof(ShaderHeader);
			totalSize += shader.vertexShader.length() * sizeof(char);
			totalSize += shader.geometryShader.length() * sizeof(char);
			totalSize += shader.fragmentShader.length() * sizeof(char);

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&header, sizeof(ShaderHeader), 1, file);
			fwrite(shader.vertexShader.c_str(), shader.vertexShader.length() * sizeof(char), 1, file);
			fwrite(shader.geometryShader.c_str(), shader.geometryShader.length() * sizeof(char), 1, file);
			fwrite(shader.fragmentShader.c_str(), shader.fragmentShader.length() * sizeof(char), 1, file);
		}	
		else{
			std::cout << "Failed to open file " << filePath << " for parsing" << std::endl;
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

		if(AssetLoader::load(filePath, texture)){
			std::vector<unsigned char> data = texture.getRawTextureData();

			Texture2DHeader header;
			header.textureId = Guid(jsonTexture["id"].ToString());
			header.width = texture.getWidth();
			header.height = texture.getHeight();
			header.numChannels = texture.getNumChannels();
			header.dimension = texture.getDimension();
			header.format = texture.getFormat();
			header.textureSize = data.size();

			int type = 1;

			size_t totalSize = sizeof(int);
			totalSize += sizeof(Texture2DHeader);
			totalSize += data.size() * sizeof(unsigned char);

			std::cout << "data size: " << data.size() << " format: " << texture.getFormat() << std::endl;
	
			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&header, sizeof(Texture2DHeader), 1, file);
			fwrite(&(data[0]), data.size() * sizeof(unsigned char), 1, file);
		}	
		else{
			std::cout << "Failed to open file " << filePath << " for parsing" << std::endl;
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

		MaterialHeader header;
		header.assetId = Guid(jsonMaterial["id"].ToString());
		header.shininess = (float)jsonMaterial["shininess"].ToFloat();
		header.ambient.x = (float)jsonMaterial["ambient"][0].ToFloat();
		header.ambient.y = (float)jsonMaterial["ambient"][1].ToFloat();
		header.ambient.z = (float)jsonMaterial["ambient"][2].ToFloat();
		header.diffuse.x = (float)jsonMaterial["diffuse"][0].ToFloat();
		header.diffuse.y = (float)jsonMaterial["diffuse"][1].ToFloat();
		header.diffuse.z = (float)jsonMaterial["diffuse"][2].ToFloat();
		header.specular.x = (float)jsonMaterial["specular"][0].ToFloat();
		header.specular.y = (float)jsonMaterial["specular"][1].ToFloat();
		header.specular.z = (float)jsonMaterial["specular"][2].ToFloat();
		header.shaderId  = Guid(jsonMaterial["shader"].ToString());
		header.textureId = Guid(jsonMaterial["mainTexture"].ToString());
		header.normalMapId = Guid(jsonMaterial["normalMap"].ToString());
		header.specularMapId = Guid(jsonMaterial["specularMap"].ToString());

		int type = 4;

		size_t totalSize = sizeof(int);
		totalSize += sizeof(MaterialHeader);

		fwrite(&totalSize, sizeof(size_t), 1, file);
		fwrite(&type, sizeof(int), 1, file);
		fwrite(&header, sizeof(MaterialHeader), 1, file);
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

		std::string filePath = meshFilePaths[i].substr(0, meshFilePaths[i].find_last_of(".")) + ".txt";

		std::cout << "mesh filepath: " << filePath << std::endl;

		if(AssetLoader::load(filePath, mesh)){
			
			// create mesh header
			MeshHeader header = {};
			header.meshId = Guid(jsonMesh["id"].ToString());
			header.verticesSize = (unsigned int)mesh.vertices.size();
			header.normalsSize = (unsigned int)mesh.normals.size();
			header.texCoordsSize = (unsigned int)mesh.texCoords.size();

			int type = 5;

			size_t totalSize = sizeof(int);
			totalSize += sizeof(MeshHeader);
			totalSize += mesh.vertices.size()*sizeof(float);
			totalSize += mesh.normals.size()*sizeof(float);
			totalSize += mesh.texCoords.size()*sizeof(float);

			std::cout << "vertices size: " << mesh.vertices.size() << " normals size: " << mesh.normals.size() << " texCoords size: " << mesh.texCoords.size() << std::endl;

			std::string outputPath = meshFilePaths[i].substr(0, meshFilePaths[i].find_last_of(".")) + ".mesh";

			fwrite(&totalSize, sizeof(size_t), 1, file);
			fwrite(&type, sizeof(int), 1, file);
			fwrite(&header, sizeof(MeshHeader), 1, file);
			fwrite(&(mesh.vertices[0]), mesh.vertices.size()*sizeof(float), 1, file);
			fwrite(&(mesh.normals[0]), mesh.normals.size()*sizeof(float), 1, file);
			fwrite(&(mesh.texCoords[0]), mesh.texCoords.size()*sizeof(float), 1, file);
		}
		else{
			std::cout << "Failed to open file " << filePath << " for parsing" << std::endl;
			return 0;
		}
	}

	// write gmeshes out to bundle
	// for(unsigned int i = 0 ; i < gmeshFilePaths.size(); i++){
	// 	// open json file and load to json object
	// 	std::ifstream in(gmeshFilePaths[i], std::ios::in | std::ios::binary);
	// 	std::ostringstream contents;
	// 	contents << in.rdbuf();
	// 	in.close();

	// 	json::JSON jsonGMesh = JSON::Load(contents.str());

	// 	GMesh gmesh;

	// 	std::string filePath = gmeshFilePaths[i].substr(0, gmeshFilePaths[i].find_last_of(".")) + ".msh";

	// 	if(AssetLoader::load(filePath, gmesh)){
			
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
	// 		header.verticesSize = gmesh.vertices.size();
	// 		header.connectSize = gmesh.connect.size();
	// 		header.bconnectSize = gmesh.bconnect.size();
	// 		header.groupsSize = gmesh.groups.size();

	// 		std::cout << "vertices size: " << gmesh.vertices.size() << " connect size: " << gmesh.connect.size() << " bconnect size: " << gmesh.bconnect.size() << " groups size: " << gmesh.groups.size() << std::endl;

	// 		std::string outputPath = gmeshFilePaths[i].substr(0, gmeshFilePaths[i].find_last_of(".")) + ".gmesh";

	// 		size = fwrite(&header, sizeof(GMeshHeader), 1, file);
	// 		size += fwrite(&gmesh.vertices[0], gmesh.vertices.size()*sizeof(float), 1, file);
	// 		size += fwrite(&gmesh.connect[0], gmesh.connect.size()*sizeof(float), 1, file);
	// 		size += fwrite(&gmesh.bconnect[0], gmesh.bconnect.size()*sizeof(float), 1, file);
	// 		size += fwrite(&gmesh.groups[0], gmesh.groups.size()*sizeof(float), 1, file);
	// 	}
	// 	else{
	// 		std::cout << "Failed to open file " << filePath << " for parsing" << std::endl;
	// 		return 0;
	// 	}
	// }

	std::cout << "Asset bundle successfully created" << std::endl;

	if(file){
		fclose(file);
	}

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