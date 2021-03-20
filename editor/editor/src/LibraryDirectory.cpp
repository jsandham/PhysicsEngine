#include "../include/LibraryDirectory.h"
#include "../include/EditorFileIO.h"
#include "../include/FileSystemUtil.h"

#include "core/Material.h"
#include "core/Mesh.h"
#include "core/Shader.h"
#include "core/Texture2D.h"

#include "components/Camera.h"
#include "components/Light.h"
#include "components/MeshRenderer.h"
#include "components/Transform.h"
#include "core/Entity.h"

#include <fstream>
#include <set>
#include <sstream>

using namespace PhysicsEditor;
using namespace PhysicsEngine;

LibraryDirectoryListener::LibraryDirectoryListener()
{
    directory = NULL;
}

void LibraryDirectoryListener::registerDirectory(LibraryDirectory *directory)
{
    this->directory = directory;
}

void LibraryDirectoryListener::handleFileAction(FW::WatchID watchid, const FW::String &dir, const FW::String &filename,
                                                FW::Action action)
{
    PhysicsEngine::Log::info(
        ("DIR (" + dir + ") FILE (" + filename + ") has event " + std::to_string(action) + "\n").c_str());

    if (FW::Action::Add || FW::Action::Modified)
    {
        directory->addToBuffer(dir + "\\" + filename);
    }
}

LibraryDirectory::LibraryDirectory()
{
    mDataPath = "";
    mWatchID = 0;
}

LibraryDirectory::~LibraryDirectory()
{
}

void LibraryDirectory::watch(const std::string& projectPath)
{
    mBuffer.clear();

    mDataPath = projectPath + "\\data";

    // register listener with library directory
    mListener.registerDirectory(this);

    // get all data files in project
    mBuffer = getFilesInDirectoryRecursive(mDataPath, true);

    // remove old watch
    mFileWatcher.removeWatch(mWatchID);

    // add watch for project data path to detect file changes
    mWatchID = mFileWatcher.addWatch(mDataPath, &mListener, true);
}

void LibraryDirectory::update()
{
    mFileWatcher.update();
}

void LibraryDirectory::addToBuffer(const std::string& filePath)
{
    mBuffer.push_back(filePath);
}

void LibraryDirectory::loadQueuedAssetsIntoWorld(PhysicsEngine::World *world)
{
    // load any assets queued up in buffer into world
    for (size_t i = 0; i < mBuffer.size(); i++)
    {
        std::string extension = getFileExtension(mBuffer[i]);

        Asset* asset = nullptr;
        if (extension == "texture") {
            asset = world->loadAssetFromYAML(mBuffer[i]);
        }
        else if (extension == "mesh")
        {
            asset = world->loadAssetFromYAML(mBuffer[i]);
        }
        else if (extension == "shader")
        {
            asset = world->loadAssetFromYAML(mBuffer[i]);
        }
        else if (extension == "material")
        {
            asset = world->loadAssetFromYAML(mBuffer[i]);
        }

        if (asset != nullptr)
        {
            filePathToId[mBuffer[i]] = asset->getId();
        }
    }

    // clear buffer
    mBuffer.clear();
}

PhysicsEngine::Guid LibraryDirectory::getFileId(const std::string &filePath) const
{
    std::map<const std::string, PhysicsEngine::Guid>::const_iterator it = filePathToId.find(filePath);
    if (it != filePathToId.end())
    {
        return it->second;
    }

    return PhysicsEngine::Guid::INVALID;
}