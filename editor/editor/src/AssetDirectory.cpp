#include "../include/AssetDirectory.h"
#include "../include/FileSystemUtil.h"

#include "core/Guid.h"
#include "json/json.hpp"

#include <fstream>

using namespace PhysicsEditor;
using namespace PhysicsEngine;

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

	std::vector<std::string> filesInProject = getFilesInDirectoryRecursive(currentProjectPath, true);
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

		std::unordered_set<std::string>::iterator it = directory.find(filesInProject[i]);
		if (it == directory.end()) {
			directory.insert(filesInProject[i]);

			// create binary version of asset in library directory
			createBinaryAssetInLibrary(filesInProject[i], extension);
		}

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
	}
}

void AssetDirectory::createBinaryAssetInLibrary(std::string filePath, std::string extension)
{
	if (extension == "mesh") {

	}
	else if (extension == "material") {

	}
	else if (extension == "png") {

	}
	else if (extension == "shader") {

	}
}