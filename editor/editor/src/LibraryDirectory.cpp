#include "../include/LibraryDirectory.h"
#include "../include/FileSystemUtil.h"
#include "../include/EditorFileIO.h"

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
#include <set>

using namespace PhysicsEditor;
using namespace PhysicsEngine;
using namespace json;

LibraryDirectory::LibraryDirectory()
{

}

LibraryDirectory::~LibraryDirectory()
{

}

void LibraryDirectory::load(std::string projectPath)
{
	currentProjectPath = projectPath;

	// if library directory doesnt exist then create it
	if (!doesDirectoryExist(currentProjectPath + "\\library")) {
		if (!createDirectory(currentProjectPath + "\\library")) {
			Log::error("Could not create library cache directory\n");
			return;
		}

		// create directory cache file
		std::ofstream file{ currentProjectPath + "\\library\\directory_cache.txt" };
	}

	libraryCache.clear();

	if (!libraryCache.load(currentProjectPath + "\\library\\directory_cache.txt")) {
		Log::error("An error occured when trying to load library. Please delete library directory so that it can be re-built\n");
	}
}

void LibraryDirectory::update(std::string projectPath)
{
	if (projectPath.empty() || projectPath != currentProjectPath) {
		return;
	}

	std::vector<FileInfo> filesToAddToLibrary;
	std::vector<std::string> filesInProject = getFilesInDirectoryRecursive(currentProjectPath + "\\data", true);

	for (size_t i = 0; i < filesInProject.size(); i++) {
		std::string extension = filesInProject[i].substr(filesInProject[i].find_last_of(".") + 1);

		if (!isFileExtensionTracked(extension)) {
			continue;
		}

		if (extension != "json") {
			std::string metaFilePath = filesInProject[i].substr(0, filesInProject[i].find_last_of(".")) + ".json";
			if (!doesFileExist(metaFilePath)) {
				createMetaFile(metaFilePath);

				filesInProject.push_back(metaFilePath);
			}
		}

		std::string createTime;
		std::string accessTime;
		std::string writeTime;

		if (!PhysicsEditor::getFileTime(filesInProject[i], createTime, accessTime, writeTime)) {
			continue; // file must be in use by another application. Trying reading later.
		}

		FileInfo fileInfo;
		fileInfo.filePath = filesInProject[i];
		fileInfo.fileExtension = extension;
		fileInfo.createTime = createTime;
		fileInfo.accessTime = accessTime;
		fileInfo.writeTime = writeTime;

		if (libraryCache.contains(filesInProject[i])) {

			if (libraryCache.isOutOfDate(filesInProject[i], createTime, writeTime)) {
				libraryCache.remove(filesInProject[i]);
				libraryCache.add(filesInProject[i], fileInfo);

				if (extension != "json") {
					filesToAddToLibrary.push_back(fileInfo);
				}
			}
		}
		else {
			libraryCache.add(filesInProject[i], fileInfo);

			if (extension != "json") {
				filesToAddToLibrary.push_back(fileInfo);
			}
		}
	}

	for (size_t i = 0; i < filesToAddToLibrary.size(); i++) {
		Guid id = findGuidFromMetaFilePath(filesToAddToLibrary[i].filePath.substr(0, filesToAddToLibrary[i].filePath.find_last_of(".")) + ".json");

		// create binary version of scene or asset in library directory
		std::string outFilePath = currentProjectPath + "\\library\\" + id.toString() + ".data";
		if (filesToAddToLibrary[i].fileExtension == "scene") {
			if (!PhysicsEditor::writeSceneToBinary(filesToAddToLibrary[i].filePath, id, outFilePath)) {
				std::string errorMessage = "An error occured when trying to create binary library version of scene: " + filesToAddToLibrary[i].filePath + "\n";
				Log::error(&errorMessage[0]);
				return;
			}
		}
		else {
			if (!PhysicsEditor::writeAssetToBinary(filesToAddToLibrary[i].filePath, filesToAddToLibrary[i].fileExtension, id, outFilePath)) {
				std::string errorMessage = "An error occured when trying to create binary library version of asset: " + filesToAddToLibrary[i].filePath + "\n";
				Log::error(&errorMessage[0]);
				return;
			}
		}
	}

	if (filesToAddToLibrary.size() > 0) {
		if (!libraryCache.save(currentProjectPath + "\\library\\directory_cache.txt")) {
			Log::error("An error occured when trying to save library\n");
		}
	}
}

LibraryCache LibraryDirectory::getLibraryCache() const
{
	return libraryCache;
}

bool LibraryDirectory::isFileExtensionTracked(std::string extension)
{
	const auto trackedExtensions = { "json", "obj", "material", "png", "shader", "scene" }; 
	for (auto ext: trackedExtensions) {
		if (extension == ext) {
			return true;
		}
	}

	return false;
}

bool LibraryDirectory::createMetaFile(std::string metaFilePath)
{
	std::fstream metaFile;

	metaFile.open(metaFilePath, std::fstream::out);

	if (metaFile.is_open()) {
		metaFile << "{\n";
		metaFile << "\t\"id\" : \"" + Guid::newGuid().toString() + "\"\n";
		metaFile << "}\n";
		metaFile.close();

		return true;
	}
	
	return false;
}

Guid LibraryDirectory::findGuidFromMetaFilePath(std::string metaFilePath)
{
	// get guid from meta file
	std::fstream metaFile;
	metaFile.open(metaFilePath, std::fstream::in);

	if (metaFile.is_open()) {
		std::ostringstream contents;
		contents << metaFile.rdbuf();

		metaFile.close();

		std::string jsonContentString = contents.str();
		json::JSON object = json::JSON::Load(contents.str());

		return object["id"].ToString();
	}
	else {
		std::string errorMessage = "An error occured when trying to open meta file: " + metaFilePath + "\n";
		Log::error(&errorMessage[0]);
		return Guid::INVALID;
	}
}