#include "../include/LibraryDirectory.h"

using namespace PhysicsEditor;

LibraryDirectoryListener::LibraryDirectoryListener()
{
    mDirectory = nullptr;
}

void LibraryDirectoryListener::registerDirectory(LibraryDirectory *directory)
{
    mDirectory = directory;
}

void LibraryDirectoryListener::handleFileAction(FW::WatchID watchid, const FW::String &dir, const FW::String &filename,
                                                FW::Action action)
{
    PhysicsEngine::Log::info(
        ("DIR (" + dir + ") FILE (" + filename + ") has event " + std::to_string(action) + "\n").c_str());

    // If file created or modified, add to buffer to load into world
    if (action == FW::Action::Add || action == FW::Action::Modified)
    {
        mDirectory->fileAddedToProject(std::filesystem::path(dir) / filename);
    }

    if (action == FW::Action::Delete)
    {
        mDirectory->fileDeletedFromProject(std::filesystem::path(dir) / filename);
    }
}

LibraryDirectory::LibraryDirectory()
{
    mDataPath = std::filesystem::path();
    mWatchID = 0;
}

LibraryDirectory::~LibraryDirectory()
{
}

void LibraryDirectory::watch(const std::filesystem::path& projectPath)
{
    mAddBuffer.clear();
    mDeleteBuffer.clear();

    mDataPath = projectPath / "data";

    // register listener with library directory
    mListener.registerDirectory(this);

    // get all data files in project
    for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(mDataPath))
    {
        if (std::filesystem::is_regular_file(entry))
        {
            mAddBuffer.push_back(entry.path());
        }
    }

    // remove old watch
    mFileWatcher.removeWatch(mWatchID);

    // add watch for project data path to detect file changes
    mWatchID = mFileWatcher.addWatch(mDataPath.string(), &mListener, true);
}

void LibraryDirectory::update(PhysicsEngine::World * world)
{
    mFileWatcher.update();

    // load any assets queued up in add buffer into world
    for (size_t i = 0; i < mAddBuffer.size(); i++)
    {
        std::string extension = mAddBuffer[i].extension().string();

        PhysicsEngine::Asset* asset = nullptr;
        if (extension == ".texture") {
            asset = world->loadAssetFromYAML(mAddBuffer[i].string());
        }
        else if (extension == ".mesh")
        {
            asset = world->loadAssetFromYAML(mAddBuffer[i].string());
        }
        else if (extension == ".shader")
        {
            asset = world->loadAssetFromYAML(mAddBuffer[i].string());
        }
        else if (extension == ".material")
        {
            asset = world->loadAssetFromYAML(mAddBuffer[i].string());
        }
        else if (extension == ".sprite")
        {
            asset = world->loadAssetFromYAML(mAddBuffer[i].string());
        }

        // ensure each png file has a generated yaml texture file and if not then create one
        if (extension == ".png")
        {
            std::string texturePath = mAddBuffer[i].string().substr(0, mAddBuffer[i].string().find_last_of(".")) + ".texture";
            if (!std::filesystem::exists(texturePath))
            {
                PhysicsEngine::Texture2D* texture = world->createAsset<PhysicsEngine::Texture2D>();
                texture->load(mAddBuffer[i].string());
                texture->writeToYAML(texturePath);

                asset = texture;
            }
        }

        // ensure each obj file has a generated yaml mesh file and if not then create one
        if (extension == ".obj")
        {
            std::string meshPath = mAddBuffer[i].string().substr(0, mAddBuffer[i].string().find_last_of(".")) + ".mesh";
            if (!std::filesystem::exists(meshPath))
            {
                PhysicsEngine::Mesh* mesh = world->createAsset<PhysicsEngine::Mesh>();
                mesh->load(mAddBuffer[i].string());
                mesh->writeToYAML(meshPath);

                asset = mesh;
            }
        }

        if (asset != nullptr)
        {
            mFilePathToId[mAddBuffer[i]] = asset->getId();
            mIdToFilePath[asset->getId()] = mAddBuffer[i];
        }
        else
        {
            PhysicsEngine::Guid fileId = PhysicsEngine::Guid::newGuid();
            mFilePathToId[mAddBuffer[i]] = fileId;
            mIdToFilePath[fileId] = mAddBuffer[i];
        }
    }

    // clear buffer
    mAddBuffer.clear();

    // destroy any assets queued up in delete buffer from world
    for (size_t i = 0; i < mDeleteBuffer.size(); i++)
    {
        std::string extension = mDeleteBuffer[i].extension().string();
        PhysicsEngine::Guid id = getId(mDeleteBuffer[i]);

        if (extension == ".texture") {
            world->immediateDestroyAsset(id, PhysicsEngine::AssetType<PhysicsEngine::Texture2D>::type);
        }
        else if (extension == ".mesh")
        {
            world->immediateDestroyAsset(id, PhysicsEngine::AssetType<PhysicsEngine::Mesh>::type);
        }
        else if (extension == ".shader")
        {
            world->immediateDestroyAsset(id, PhysicsEngine::AssetType<PhysicsEngine::Shader>::type);
        }
        else if (extension == ".material")
        {
            world->immediateDestroyAsset(id, PhysicsEngine::AssetType<PhysicsEngine::Material>::type);
        }
        else if (extension == ".sprite")
        {
            world->immediateDestroyAsset(id, PhysicsEngine::AssetType<PhysicsEngine::Sprite>::type);
        }

        mFilePathToId.erase(mDeleteBuffer[i]);
        mIdToFilePath.erase(id);
    }

    // clear buffer
    mDeleteBuffer.clear();
}

void LibraryDirectory::fileAddedToProject(const std::filesystem::path& filePath)
{
    mAddBuffer.push_back(filePath);
}

void LibraryDirectory::fileDeletedFromProject(const std::filesystem::path& filePath)
{
    mDeleteBuffer.push_back(filePath);
}

PhysicsEngine::Guid LibraryDirectory::getId(const std::filesystem::path& filePath) const
{
    std::map<const std::filesystem::path, PhysicsEngine::Guid>::const_iterator it = mFilePathToId.find(filePath);
    if (it != mFilePathToId.end())
    {
        return it->second;
    }

    return PhysicsEngine::Guid::INVALID;
}

std::filesystem::path LibraryDirectory::getFile(const PhysicsEngine::Guid& id) const
{
    std::map<const PhysicsEngine::Guid, std::filesystem::path>::const_iterator it = mIdToFilePath.find(id);
    if (it != mIdToFilePath.end())
    {
        return it->second;
    }

    return std::filesystem::path();
}