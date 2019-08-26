#include "../include/AssetDirectory.h"
#include "../include/FileSystemUtil.h"

#include "core/Guid.h"
#include "core/AssetLoader.h"
#include "core/Texture2D.h"
#include "core/Shader.h"
#include "core/Mesh.h"
#include "core/Material.h"
#include "json/json.hpp"

#include <fstream>
#include <sstream>

using namespace PhysicsEditor;
using namespace PhysicsEngine;
using namespace json;

AssetDirectory::AssetDirectory()
{

}

AssetDirectory::~AssetDirectory()
{

}

void AssetDirectory::update(std::string projectPath)
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

		directory.clear();
	}

	const std::string trackedExtensions[] = { "mesh", "material", "png", "shader" };

	std::vector<std::string> filesInProject = getFilesInDirectoryRecursive(currentProjectPath + "\\data", true);
	for (size_t i = 0; i < filesInProject.size(); i++) {
		std::string extension = filesInProject[i].substr(filesInProject[i].find_last_of(".") + 1);

		bool isValid = false;
		for (int j = 0; j < 4; j++) {
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

		std::unordered_set<std::string>::iterator it = directory.find(filesInProject[i]);
		if (it == directory.end()) {
			directory.insert(filesInProject[i]);

			// create binary version of asset in library directory
			createBinaryAssetInLibrary(filesInProject[i], extension);
		}
	}
}

void AssetDirectory::createBinaryAssetInLibrary(std::string filePath, std::string extension)
{
	std::string metaFilePath = filePath.substr(0, filePath.find(".")) + ".json";

	// get guid from asset meta file
	std::ifstream metaFile(metaFilePath, std::ios::in | std::ios::binary);

	Guid guid;
	if (metaFile.is_open()) {
		std::ostringstream contents;
		contents << metaFile.rdbuf();
		metaFile.close();

		json::JSON jsonObject = JSON::Load(contents.str());
		guid = Guid(jsonObject["id"].ToString());
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
	else if (extension == "mesh") {
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
	std::string filename = filePath.substr(filePath.find_last_of("/\\") + 1);
	std::fstream file(currentProjectPath + "\\library\\" + filename, std::ios::out | std::ios::binary);

	if (file.is_open()) {
		char classification = 'a';
		size_t size = data.size();

		file.write(&classification, 1);
		file.write((char*)&assetType, sizeof(int));
		file.write((char*)&size, sizeof(size_t));
		file.write(&data[0], data.size());

		file.close();
	}
}