#include "../include/LibraryDirectory.h"
#include "../include/FileSystemUtil.h"
#include "../include/EditorFileIO.h"

#include "core/Scene.h"
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

LibraryDirectoryListener::LibraryDirectoryListener()
{
	directory = NULL;
}

void LibraryDirectoryListener::registerDirectory(LibraryDirectory* directory)
{
	this->directory = directory;
}

void LibraryDirectoryListener::handleFileAction(FW::WatchID watchid, const FW::String& dir, const FW::String& filename, FW::Action action)
{
	PhysicsEngine::Log::info(("DIR (" + dir + ") FILE (" + filename + ") has event " + std::to_string(action) + "\n").c_str());

	if (FW::Action::Add || FW::Action::Modified) {
		directory->generateBinaryLibraryFile(dir + "\\" + filename);
	}
}

LibraryDirectory::LibraryDirectory()
{
	mDataPath = "";
	mLibraryPath = "";
	mWatchID = 0;
}

LibraryDirectory::~LibraryDirectory()
{

}

void LibraryDirectory::watch(std::string projectPath)
{
	mBuffer.clear();

	mDataPath = projectPath + "\\data";
	mLibraryPath = projectPath + "\\library";

	// register listener with library directory
	mListener.registerDirectory(this);

	// if library directory doesnt exist then create it
	if (!doesDirectoryExist(mLibraryPath)) {
		if (!createDirectory(mLibraryPath)) {
			Log::error("Could not create library cache directory\n");
			return;
		}
	}

	// get all data files in project
	std::vector<std::string>& filesInProject = getFilesInDirectoryRecursive(mDataPath, true);

	// generate binary library file from project data file
	for (auto file : filesInProject){
		generateBinaryLibraryFile(file);
	}

	// remove old watch
	mFileWatcher.removeWatch(mWatchID);

	// add watch for project data path to detect file changes
	mWatchID = mFileWatcher.addWatch(mDataPath, &mListener, true);
}

void LibraryDirectory::update()
{
	mFileWatcher.update();
}

void LibraryDirectory::generateBinaryLibraryFile(std::string filePath)
{
	std::string extension = getFileExtension(filePath);

	if (extension != "json") {
		std::string metaFilePath = filePath.substr(0, filePath.find_last_of(".")) + ".json";

		if (!doesFileExist(metaFilePath)) {
			if (!createMetaFile(metaFilePath)){
				std::string errorMessage = "Could not create meta file " + metaFilePath + "\n";
				PhysicsEngine::Log::error(errorMessage.c_str());
				return;
			}
		}

		PhysicsEngine::Guid id = findGuidFromMetaFilePath(metaFilePath);

		filePathToId[filePath] = id;

		// create binary version of scene or asset in library directory
		std::string binaryFilePath = mLibraryPath + "\\" + id.toString();

		bool success = false;

		if (extension == "scene") {
			binaryFilePath += ".sdata";
			success = PhysicsEditor::writeSceneToBinary(filePath, id, binaryFilePath);
		}
		else if(extension == "material" || extension == "obj" || extension == "shader" || extension == "png"){
			binaryFilePath += ".data";
			success = PhysicsEditor::writeAssetToBinary(filePath, extension, id, binaryFilePath);
		}

		if (!success) {
			std::string errorMessage = "An error occured when trying to create binary library version of data: " + filePath + "\n";
			PhysicsEngine::Log::error(errorMessage.c_str());
			return;
		}

		mBuffer.push_back(binaryFilePath);
	}
}

void LibraryDirectory::loadQueuedAssetsIntoWorld(PhysicsEngine::World* world)
{
	// load any assets queued up in buffer into world
	for (size_t i = 0; i < mBuffer.size(); i++) {
		if (getFileExtension(mBuffer[i]) == "data") {
			if (!world->loadAsset(mBuffer[i])) {
				std::string errorMessage = "Could not load asset: " + mBuffer[i] + "\n";
				Log::error(errorMessage.c_str());
			}
		}
	}

	// clear buffer
	mBuffer.clear();
}

PhysicsEngine::Guid LibraryDirectory::getFileId(const std::string& filePath) const
{
	std::map<const std::string, PhysicsEngine::Guid>::const_iterator it = filePathToId.find(filePath);
	if (it != filePathToId.end())
	{
		return it->second;
	}

	return PhysicsEngine::Guid::INVALID;
}