#include <fstream>
#include <set>
#include <sstream>

#include "../include/LibraryDirectory.h"
#include "../include/FileSystemUtil.h"

using namespace PhysicsEditor;

LibraryDirectoryListener::LibraryDirectoryListener()
{
    directory = nullptr;
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

    // If file created or modified, add to buffer to load into world
    if (action == FW::Action::Add || action == FW::Action::Modified)
    {
        directory->fileAddedToProject(dir + "\\" + filename);
    }

    if (action == FW::Action::Delete)
    {
        directory->fileDeletedFromProject(dir + "\\" + filename);
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
    mAddBuffer.clear();
    mDeleteBuffer.clear();

    mDataPath = projectPath + "\\data";

    // register listener with library directory
    mListener.registerDirectory(this);

    // get all data files in project
    mAddBuffer = getFilesInDirectoryRecursive(mDataPath, true);

    // remove old watch
    mFileWatcher.removeWatch(mWatchID);

    // add watch for project data path to detect file changes
    mWatchID = mFileWatcher.addWatch(mDataPath, &mListener, true);
}

void LibraryDirectory::update(PhysicsEngine::World * world)
{
    mFileWatcher.update();

    // load any assets queued up in add buffer into world
    for (size_t i = 0; i < mAddBuffer.size(); i++)
    {
        std::string extension = getFileExtension(mAddBuffer[i]);

        PhysicsEngine::Asset* asset = nullptr;
        if (extension == "texture") {
            asset = world->loadAssetFromYAML(mAddBuffer[i]);
        }
        else if (extension == "mesh")
        {
            asset = world->loadAssetFromYAML(mAddBuffer[i]);
        }
        else if (extension == "shader")
        {
            asset = world->loadAssetFromYAML(mAddBuffer[i]);
        }
        else if (extension == "material")
        {
            asset = world->loadAssetFromYAML(mAddBuffer[i]);
        }

        if (asset != nullptr)
        {
            mFilePathToId[mAddBuffer[i]] = asset->getId();
            mIdToFilePath[asset->getId()] = mAddBuffer[i];
        }
    }

    // clear buffer
    mAddBuffer.clear();

    // destroy any assets queued up in delete buffer from world
    for (size_t i = 0; i < mDeleteBuffer.size(); i++)
    {
        std::string extension = getFileExtension(mDeleteBuffer[i]);
        PhysicsEngine::Guid id = getId(mDeleteBuffer[i]);

        if (extension == "texture") {
            world->immediateDestroyAsset(id, PhysicsEngine::AssetType<PhysicsEngine::Texture2D>::type);
        }
        else if (extension == "mesh")
        {
            world->immediateDestroyAsset(id, PhysicsEngine::AssetType<PhysicsEngine::Mesh>::type);
        }
        else if (extension == "shader")
        {
            world->immediateDestroyAsset(id, PhysicsEngine::AssetType<PhysicsEngine::Shader>::type);
        }
        else if (extension == "material")
        {
            world->immediateDestroyAsset(id, PhysicsEngine::AssetType<PhysicsEngine::Material>::type);
        }

        mFilePathToId.erase(mDeleteBuffer[i]);
        mIdToFilePath.erase(id);
    }

    // clear buffer
    mDeleteBuffer.clear();
}

void LibraryDirectory::fileAddedToProject(const std::string& filePath)
{
    mAddBuffer.push_back(filePath);
}

void LibraryDirectory::fileDeletedFromProject(const std::string& filePath)
{
    mDeleteBuffer.push_back(filePath);
}

PhysicsEngine::Guid LibraryDirectory::getId(const std::string &filePath) const
{
    std::map<const std::string, PhysicsEngine::Guid>::const_iterator it = mFilePathToId.find(filePath);
    if (it != mFilePathToId.end())
    {
        return it->second;
    }

    return PhysicsEngine::Guid::INVALID;
}

std::string LibraryDirectory::getFile(const PhysicsEngine::Guid& id) const
{
    std::map<const PhysicsEngine::Guid, std::string>::const_iterator it = mIdToFilePath.find(id);
    if (it != mIdToFilePath.end())
    {
        return it->second;
    }

    return "";
}