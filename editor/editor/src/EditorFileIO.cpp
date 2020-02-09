#include <fstream>
#include <sstream>

#include "../include/EditorFileIO.h"

#include "core/AssetLoader.h"
#include "core/Log.h"
#include "core/Entity.h"

#include "json/json.hpp"

using namespace PhysicsEditor;
using namespace PhysicsEngine;
using namespace json;

bool PhysicsEditor::writeAssetToBinary(std::string filePath, std::string fileExtension, Guid id, std::string outFilePath)
{
	std::string infoMessage = "Writing binary version of asset " + filePath + " to library\n";
	Log::info(&infoMessage[0]);

	// load data from asset
	std::vector<char> data;
	int assetType = -1;
	if (fileExtension == "shader") {
		assetType = AssetType<Shader>::type;
		Shader shader;

		if (AssetLoader::load(filePath, shader)) {
			shader.assetId = id.toString();
			data = shader.serialize();
		}
	}
	else if (fileExtension == "png") {
		assetType = AssetType<Texture2D>::type;
		Texture2D texture;

		if (AssetLoader::load(filePath, texture)) {
			texture.assetId = id.toString();
			data = texture.serialize();
		}
	}
	else if (fileExtension == "obj") {
		assetType = AssetType<Mesh>::type;
		Mesh mesh;

		if (AssetLoader::load(filePath, mesh)) {
			mesh.assetId = id.toString();
			data = mesh.serialize();
		}
	}
	else if (fileExtension == "material") {
		assetType = AssetType<Material>::type;
		Material material;

		if (AssetLoader::load(filePath, material)) {
			material.assetId = id.toString();
			data = material.serialize();
		}
	}

	// write data to binary version of asset in library 
	std::fstream outFile(outFilePath, std::ios::out | std::ios::binary);

	if (outFile.is_open()) {
		AssetHeader header = {};

		outFile.write((char*)&header, sizeof(header));

		char classification = 'a';
		size_t size = data.size();

		outFile.write(&classification, 1);
		outFile.write((char*)&assetType, sizeof(int));
		outFile.write((char*)&size, sizeof(size_t));
		outFile.write(&data[0], data.size());

		outFile.close();
	}
	else {
		std::string errorMessage = "Could not open file " + outFilePath + " for writing to library\n";
		Log::error(&errorMessage[0]);
		return false;
	}

	return true;
}

bool PhysicsEditor::writeSceneToBinary(std::string filePath, Guid id, std::string outFilePath)
{
	std::string infoMessage = "Writing binary version of scene " + filePath + " to library\n";
	Log::info(&infoMessage[0]);

	std::fstream file;

	file.open(filePath);

	std::ostringstream contents;
	if (file.is_open())
	{
		contents << file.rdbuf();
		file.close();
	}
	else {
		std::string errorMessage = "Could not open scene " + filePath + " for writing to library\n";
		Log::error(&errorMessage[0]);
		return false;
	}

	std::string contentString = contents.str();
	json::JSON jsonScene = json::JSON::Load(contentString);

	// parse loaded json file
	json::JSON entities;
	json::JSON transforms;
	json::JSON cameras;
	json::JSON meshRenderers;
	json::JSON lights;
	json::JSON boxColliders;
	json::JSON sphereColliders;

	json::JSON::JSONWrapper<map<string, JSON>> objects = jsonScene.ObjectRange();
	map<string, JSON>::iterator it;

	for (it = objects.begin(); it != objects.end(); it++) {
		std::string type = it->second["type"].ToString();

		if (type == "Entity") {
			entities[it->first] = it->second;
		}
		else if (type == "Transform") {
			transforms[it->first] = it->second;
		}
		else if (type == "Camera") {
			cameras[it->first] = it->second;
		}
		else if (type == "MeshRenderer") {
			meshRenderers[it->first] = it->second;
		}
		else if (type == "Light") {
			lights[it->first] = it->second;
		}
		else if (type == "BoxCollider") {
			boxColliders[it->first] = it->second;
		}
		else if(type == "SphereCollider") {
			sphereColliders[it->first] = it->second;
		}
	}

	std::fstream outFile(outFilePath, std::ios::out | std::ios::binary);

	if (outFile.is_open()) {
		SceneHeader header = {};

		outFile.write((char*)&header, sizeof(header));

		// serialize entities
		if (!entities.IsNull()) {
			json::JSON::JSONWrapper<map<string, JSON>> entityObjects = entities.ObjectRange();
			for (it = entityObjects.begin(); it != entityObjects.end(); it++) {
				Entity entity;
				entity.entityId = Guid(it->first);
				entity.doNotDestroy = false;

				std::vector<char> data = entity.serialize();

				char classification = 'e';
				int type = 0;
				size_t size = data.size();

				outFile.write(&classification, 1);
				outFile.write((char*)&type, sizeof(int));
				outFile.write((char*)&size, sizeof(size_t));
				outFile.write(&data[0], data.size());
			}
		}

		// serialize transforms
		if (!transforms.IsNull()) {
			json::JSON::JSONWrapper<map<string, JSON>> transformObjects = transforms.ObjectRange();
			for (it = transformObjects.begin(); it != transformObjects.end(); it++) {
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

				outFile.write(&classification, 1);
				outFile.write((char*)&type, sizeof(int));
				outFile.write((char*)&size, sizeof(size_t));
				outFile.write(&data[0], data.size());
			}
		}

		// serialize camera
		if (!cameras.IsNull()) {
			json::JSON::JSONWrapper<map<string, JSON>> cameraObjects = cameras.ObjectRange();
			for (it = cameraObjects.begin(); it != cameraObjects.end(); it++) {
				Camera camera;

				camera.componentId = Guid(it->first);
				camera.entityId = Guid(it->second["entity"].ToString());
				camera.targetTextureId = Guid(it->second["targetTextureId"].ToString());

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

				outFile.write(&classification, 1);
				outFile.write((char*)&type, sizeof(int));
				outFile.write((char*)&size, sizeof(size_t));
				outFile.write(&data[0], data.size());
			}
		}

		// serialize mesh renderers
		if (!meshRenderers.IsNull()) {
			json::JSON::JSONWrapper<map<string, JSON>> meshRendererObjects = meshRenderers.ObjectRange();
			for (it = meshRendererObjects.begin(); it != meshRendererObjects.end(); it++) {
				MeshRenderer meshRenderer;

				meshRenderer.componentId = Guid(it->first);
				meshRenderer.entityId = Guid(it->second["entity"].ToString());
				meshRenderer.meshId = Guid(it->second["mesh"].ToString());

				if (it->second.hasKey("material")) {
					meshRenderer.materialCount = 1;
					meshRenderer.materialIds[0] = Guid(it->second["material"].ToString());
					for (int j = 1; j < 8; j++) {
						meshRenderer.materialIds[j] = Guid::INVALID;
					}
				}
				else if (it->second.hasKey("materials")) {
					int materialCount = it->second["materials"].length();
					if (materialCount > 8) {
						Log::error("Currently only support at most 8 materials");
						return false;
					}

					meshRenderer.materialCount = materialCount;

					for (int j = 0; j < materialCount; j++) {
						meshRenderer.materialIds[j] = Guid(it->second["materials"][j].ToString());
					}

					for (int j = materialCount; j < 8; j++) {
						meshRenderer.materialIds[j] = Guid::INVALID;
					}
				}

				meshRenderer.isStatic = it->second["isStatic"].ToBool();

				std::vector<char> data = meshRenderer.serialize();

				char classification = 'c';
				int type = 3;
				size_t size = data.size();

				outFile.write(&classification, 1);
				outFile.write((char*)&type, sizeof(int));
				outFile.write((char*)&size, sizeof(size_t));
				outFile.write(&data[0], data.size());
			}
		}

		// serialize lights
		if (!lights.IsNull()) {
			json::JSON::JSONWrapper<map<string, JSON>> lightObjects = lights.ObjectRange();
			for (it = lightObjects.begin(); it != lightObjects.end(); it++) {
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

				outFile.write(&classification, 1);
				outFile.write((char*)&type, sizeof(int));
				outFile.write((char*)&size, sizeof(size_t));
				outFile.write(&data[0], data.size());
			}
		}
		if (!boxColliders.IsNull()) {
			json::JSON::JSONWrapper<map<string, JSON>> boxColliderObjects = boxColliders.ObjectRange();
			for (it = boxColliderObjects.begin(); it != boxColliderObjects.end(); it++) {
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

				outFile.write(&classification, 1);
				outFile.write((char*)&type, sizeof(int));
				outFile.write((char*)&size, sizeof(size_t));
				outFile.write(&data[0], data.size());
			}

		}
		if (!sphereColliders.IsNull()) {
			json::JSON::JSONWrapper<map<string, JSON>> sphereColliderObjects = sphereColliders.ObjectRange();
			for (it = sphereColliderObjects.begin(); it != sphereColliderObjects.end(); it++) {
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

				outFile.write(&classification, 1);
				outFile.write((char*)&type, sizeof(int));
				outFile.write((char*)&size, sizeof(size_t));
				outFile.write(&data[0], data.size());

			}

		}

		outFile.close();
	}
	else {
		std::string errorMessage = "Could not open file " + outFilePath + " for writing\n";
		Log::error(&errorMessage[0]);
		return false;
	}

	return true;
}

bool PhysicsEditor::writeWorldToJson(PhysicsEngine::World* world, std::string outFilePath)
{
	std::ofstream file;

	std::string test = "Writing world to file " + outFilePath + "\n";
	PhysicsEngine::Log::info(test.c_str());

	file.open(outFilePath, std::ios::out);

	if (!file.is_open()) {
		std::string message = "Could not write world to file path " + outFilePath + "\n";
		PhysicsEngine::Log::error(message.c_str());
		return false;
	}

	json::JSON obj;

	for (int i = 0; i < world->getNumberOfEntities(); i++) {
		Entity* entity = world->getEntityByIndex(i);

		if (entity->entityId == Guid("11111111-1111-1111-1111-111111111111")) {
			continue;
		}

		std::vector<std::pair<Guid, int>> componentsOnEntity = entity->getComponentsOnEntity(world);

		Guid entityId = entity->entityId;

		// write entity to json
		obj[entityId.toString()] = json::Object();
		obj[entityId.toString()]["type"] = "Entity";
		for (size_t j = 0; j < componentsOnEntity.size(); j++) {
			obj[entityId.toString()]["components"].append(componentsOnEntity[j].first.toString());
		}

		// write each component on entity to json
		for (size_t j = 0; j < componentsOnEntity.size(); j++) {
			Guid componentId = componentsOnEntity[j].first;
			int componentType = componentsOnEntity[j].second;

			//json::JSON componentObj = json::Object();
			if (componentType == 0) {
				Transform* transform = world->getComponent<Transform>(entityId);
				
				obj[componentId.toString()]["type"] = "Transform";
				obj[componentId.toString()]["parent"] = transform->parentId.toString();
				obj[componentId.toString()]["entity"] = entityId.toString();
				obj[componentId.toString()]["position"].append(transform->position.x, transform->position.y, transform->position.z);
				obj[componentId.toString()]["rotation"].append(transform->rotation.x, transform->rotation.y, transform->rotation.z, transform->rotation.w);
				obj[componentId.toString()]["scale"].append(transform->scale.x, transform->scale.y, transform->scale.z);
			}
			else if (componentType == 1) {
				Rigidbody* rigidbody = world->getComponent<Rigidbody>(entityId);
			}
			else if (componentType == 2) {
				Camera* camera = world->getComponent<Camera>(entityId);

				obj[componentId.toString()]["type"] = "Camera";
				obj[componentId.toString()]["entity"] = entityId.toString();
				obj[componentId.toString()]["targetTextureId"] = camera->targetTextureId.toString();
				obj[componentId.toString()]["position"].append(camera->position.x, camera->position.y, camera->position.z);
				obj[componentId.toString()]["front"].append(camera->front.x, camera->front.y, camera->front.z);
				obj[componentId.toString()]["up"].append(camera->up.x, camera->up.y, camera->up.z);
				obj[componentId.toString()]["backgroundColor"].append(camera->backgroundColor.x, camera->backgroundColor.y, camera->backgroundColor.z, camera->backgroundColor.w);
				obj[componentId.toString()]["x"] = camera->viewport.x;
				obj[componentId.toString()]["y"] = camera->viewport.y;
				obj[componentId.toString()]["width"] = camera->viewport.width;
				obj[componentId.toString()]["height"] = camera->viewport.height;
				obj[componentId.toString()]["fov"] = camera->frustum.fov;
				obj[componentId.toString()]["near"] = camera->frustum.nearPlane;
				obj[componentId.toString()]["far"] = camera->frustum.farPlane;
			}
			else if (componentType == 3) {
				MeshRenderer* meshRenderer = world->getComponent<MeshRenderer>(entityId);

				obj[componentId.toString()]["type"] = "MeshRenderer";
				obj[componentId.toString()]["entity"] = entityId.toString();
				obj[componentId.toString()]["mesh"] = meshRenderer->meshId.toString();

				int materialCount = meshRenderer->materialCount;

				std::string label = "material";
				if (materialCount > 1) {
					label = "materials";
				}

				std::string value = "";
				if (materialCount == 0) {
					value = Guid::INVALID.toString();
				}
				else if (materialCount == 1) {
					value = meshRenderer->materialIds[0].toString();
				}
				else { // dont think this is right. I think I need to do something like obj[componentId.toString()][label].append...
					value += "[";
					for (int m = 0; m < materialCount; m++) {
						value += meshRenderer->materialIds[i].toString();
						if (m != materialCount - 1) {
							value += ",";
						}
					}
					value += "]";
				}
			
				obj[componentId.toString()][label] = value;
				obj[componentId.toString()]["isStatic"] = meshRenderer->isStatic;
			}
			else if (componentType == 4) {
				LineRenderer* lineRenderer = world->getComponent<LineRenderer>(entityId);
			}
			else if (componentType == 5) {
				Light* light = world->getComponent<Light>(entityId);

				obj[componentId.toString()]["type"] = "Light";
				obj[componentId.toString()]["entity"] = entityId.toString();
				obj[componentId.toString()]["position"].append(light->position.x, light->position.y, light->position.z);
				obj[componentId.toString()]["direction"].append(light->direction.x, light->direction.y, light->direction.z);
				obj[componentId.toString()]["ambient"].append(light->ambient.x, light->ambient.y, light->ambient.z);
				obj[componentId.toString()]["diffuse"].append(light->diffuse.x, light->diffuse.y, light->diffuse.z);
				obj[componentId.toString()]["specular"].append(light->specular.x, light->specular.y, light->specular.z);
				obj[componentId.toString()]["constant"] = light->constant;
				obj[componentId.toString()]["linear"] = light->linear;
				obj[componentId.toString()]["quadratic"] = light->quadratic;
				obj[componentId.toString()]["cutOff"] = light->cutOff;
				obj[componentId.toString()]["outerCutOff"] = light->outerCutOff;
				obj[componentId.toString()]["lightType"] = static_cast<int>(light->lightType);
				obj[componentId.toString()]["shadowType"] = static_cast<int>(light->shadowType);
			}
			else if (componentType == 8) {
				BoxCollider* collider = world->getComponent<BoxCollider>(entityId);

				obj[componentId.toString()]["type"] = "SphereCollider";
				obj[componentId.toString()]["entity"] = entityId.toString();

				obj[componentId.toString()]["centre"].append(collider->bounds.centre.x, collider->bounds.centre.y, collider->bounds.centre.z);
				obj[componentId.toString()]["size"].append(collider->bounds.size.x, collider->bounds.size.y, collider->bounds.size.z);
			}
			else if (componentType == 9) {
				SphereCollider* collider = world->getComponent<SphereCollider>(entityId);

				obj[componentId.toString()]["type"] = "SphereCollider";
				obj[componentId.toString()]["entity"] = entityId.toString();

				obj[componentId.toString()]["centre"].append(collider->sphere.centre.x, collider->sphere.centre.y, collider->sphere.centre.z);
				obj[componentId.toString()]["radius"] = collider->sphere.radius;
			}
			else if (componentType == 10) {
				CapsuleCollider* capsuleCollider = world->getComponent<CapsuleCollider>(entityId);
			}
			else if (componentType == 15) {
				MeshCollider* meshCollider = world->getComponent<MeshCollider>(entityId);
			}
		}



		//// Create a new Array as a field of an Object.
		//obj["array"] = json::Array(true, "Two", 3, 4.0);
		//// Create a new Object as a field of another Object.
		//obj["obj"] = json::Object();
		//// Assign to one of the inner object's fields
		//obj["obj"]["inner"] = "Inside";

		//// We don't need to specify the type of the JSON object:
		//obj["new"]["some"]["deep"]["key"] = "Value";
		//obj["array2"].append(false, "three");

		//// We can also parse a string into a JSON object:
		//obj["parsed"] = JSON::Load("[ { \"Key\" : \"Value\" }, false ]");
	}

	file << obj;
	file << "\n";

	file.close();

	return true;
}