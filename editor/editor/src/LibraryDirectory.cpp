#include "../include/LibraryDirectory.h"
#include "../include/FileSystemUtil.h"

#include "core/Log.h"
#include "core/Scene.h"
#include "core/AssetLoader.h"
#include "core/Texture2D.h"
#include "core/Shader.h"
#include "core/Mesh.h"
#include "core/Material.h"

#include "core/Entity.h"
#include "components/Transform.h"
#include "components/MeshRenderer.h"
#include "components/Light.h"
#include "components/Camera.h"

#include "json/json.hpp"

#include <fstream>
#include <sstream>

using namespace PhysicsEditor;
using namespace PhysicsEngine;
using namespace json;

LibraryDirectory::LibraryDirectory()
{

}

LibraryDirectory::~LibraryDirectory()
{

}

void LibraryDirectory::update(std::string projectPath)
{
	if (projectPath.empty()) {
		return;
	}

	if (projectPath != currentProjectPath) {
		currentProjectPath = projectPath;

		// create library directory if it does not exist, or clear it if it does
		if (doesDirectoryExist(currentProjectPath + "\\library")) {
			deleteDirectory(currentProjectPath + "\\library");
		}

		createDirectory(currentProjectPath + "\\library");

		trackedFilesInProject.clear();
	}

	const std::string trackedExtensions[] = { "obj", "material", "png", "shader", "scene" };

	std::vector<std::string> filesInProject = getFilesInDirectoryRecursive(currentProjectPath + "\\data", true);
	for (size_t i = 0; i < filesInProject.size(); i++) {
		std::string extension = filesInProject[i].substr(filesInProject[i].find_last_of(".") + 1);

		bool isValid = false;
		for (int j = 0; j < 5; j++) {
			if (extension == trackedExtensions[j]) {
				isValid = true;
				break;
			}
		}

		if (!isValid) { continue; }

		std::string metaFilename = filesInProject[i].substr(0, filesInProject[i].find_last_of(".")) + ".json";
		if (!doesFileExist(metaFilename)) {
			std::fstream file;

			file.open(metaFilename, std::fstream::out);

			if (file.is_open()) {
				file << "{\n";
				file << "\t\"id\" : \"" + Guid::newGuid().toString() + "\"\n";
				file << "}\n";
				file.close();
			}
		}
		
		// if file is not tracked, then add it to library
		std::unordered_set<std::string>::iterator it = trackedFilesInProject.find(filesInProject[i]);
		if (it == trackedFilesInProject.end()) {
			trackedFilesInProject.insert(filesInProject[i]);

			// get guid from meta file
			std::fstream metaFile;
			metaFile.open(metaFilename, std::fstream::in);

			if (metaFile.is_open()) {
				std::ostringstream contents;
				contents << metaFile.rdbuf();

				metaFile.close();

				std::string jsonContentString = contents.str();
				json::JSON object = json::JSON::Load(contents.str());

				Guid id = object["id"].ToString();

				std::map<Guid, std::string>::iterator it1 = idToTrackedFilePath.find(id);
				if (it1 != idToTrackedFilePath.end()) {
					idToTrackedFilePath[id] = filesInProject[i];
				}
			}

			// create binary version of scene or asset in library directory
			if (extension == "scene"){
				if (!createBinarySceneInLibrary(filesInProject[i])) {
					std::string errorMessage = "An error occured when trying to create binary library version of scene: " + filesInProject[i] + "\n";
					Log::error(&errorMessage[0]);
					return;
				}
			}
			else {
				if (!createBinaryAssetInLibrary(filesInProject[i], extension)){
					std::string errorMessage = "An error occured when trying to create binary library version of asset: " + filesInProject[i] + "\n";
					Log::error(&errorMessage[0]);
					return;
				}
			}
		}
	}
}

std::unordered_set<std::string> LibraryDirectory::getTrackedFilesInProject() const
{
	return trackedFilesInProject;
}

std::string LibraryDirectory::getPathToBinarySceneOrAsset(Guid id)
{
	std::map<Guid, std::string>::iterator it = idToTrackedFilePath.find(id);
	if (it != idToTrackedFilePath.end()){
		return it->second;
	}

	return std::string("");
}

bool LibraryDirectory::createBinaryAssetInLibrary(std::string filePath, std::string extension)
{
	std::string metaFilePath = filePath.substr(0, filePath.find(".")) + ".json";

	// get guid from asset meta file
	std::ifstream metaFile(metaFilePath, std::ios::in);

	Guid guid;
	if (metaFile.is_open()) {
		std::ostringstream contents;
		contents << metaFile.rdbuf();
		metaFile.close();

		json::JSON jsonObject = JSON::Load(contents.str());
		guid = Guid(jsonObject["id"].ToString());
	}
	else {
		std::string errorMessage = "Could not open meta file " + metaFilePath + "\n";
		Log::error(&errorMessage[0]);
		return false;
	}

	// load data from asset
	std::vector<char> data;
	int assetType = -1;
	if (extension == "shader") {
		assetType = AssetType<Shader>::type;
		Shader shader;

		if (AssetLoader::load(filePath, shader)) {
			shader.assetId = guid.toString();
			data = shader.serialize();
		}
	}
	else if (extension == "png") {
		assetType = AssetType<Texture2D>::type;
		Texture2D texture;

		if (AssetLoader::load(filePath, texture)) {
			texture.assetId = guid.toString();
			data = texture.serialize();
		}
	}
	else if (extension == "obj") {
		assetType = AssetType<Mesh>::type;
		Mesh mesh;

		if (AssetLoader::load(filePath, mesh)) {
			mesh.assetId = guid.toString();
			data = mesh.serialize();
		}
	}
	else if (extension == "material") {
		assetType = AssetType<Material>::type;
		Material material;

		if (AssetLoader::load(filePath, material)) {
			material.assetId = guid.toString();
			data = material.serialize();
		}
	}

	// write data to binary version of asset in library 
	std::string temp = filePath.substr(filePath.find_last_of("\\") + 1);
	std::string outFilename = temp.substr(0, temp.find_last_of(".")) + ".data";
	std::string outFilePath = currentProjectPath + "\\library\\" + outFilename;
	std::fstream outFile(outFilePath, std::ios::out | std::ios::binary);

	if (outFile.is_open()) {
		AssetHeader header = {};

		outFile.write((char*)& header, sizeof(header));

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

bool LibraryDirectory::createBinarySceneInLibrary(std::string filePath)
{
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
	}

	std::string temp = filePath.substr(filePath.find_last_of("\\") + 1);
	std::string outFilename = temp.substr(0, temp.find_last_of(".")) + ".data";
	std::string outFilePath = currentProjectPath + "\\library\\" + outFilename;
	std::fstream outFile(outFilePath, std::ios::out | std::ios::binary);

	if (outFile.is_open()) {
		SceneHeader header = {};

		outFile.write((char*)& header, sizeof(header));

		// serialize entities
		if (!entities.IsNull()) {
			json::JSON::JSONWrapper<map<string, JSON>> entityObjects = entities.ObjectRange();
			for (it = entityObjects.begin(); it != entityObjects.end(); it++) {
				Entity entity;
				entity.entityId = Guid(it->first);

				std::vector<char> data = entity.serialize();

				char classification = 'e';
				int type = 0;
				size_t size = data.size();

				outFile.write(&classification, 1);
				outFile.write((char*)& type, sizeof(int));
				outFile.write((char*)& size, sizeof(size_t));
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
				outFile.write((char*)& type, sizeof(int));
				outFile.write((char*)& size, sizeof(size_t));
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
				outFile.write((char*)& type, sizeof(int));
				outFile.write((char*)& size, sizeof(size_t));
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
				outFile.write((char*)& type, sizeof(int));
				outFile.write((char*)& size, sizeof(size_t));
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
				outFile.write((char*)& type, sizeof(int));
				outFile.write((char*)& size, sizeof(size_t));
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