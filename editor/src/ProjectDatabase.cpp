#include "../include/ProjectDatabase.h"

#include <fstream>

using namespace PhysicsEditor;

void DirectoryListener::handleFileAction(efsw::WatchID watchid, const std::string& dir, const std::string& filename, efsw::Action action, std::string oldFilename) 
{
    switch (action) {
    case efsw::Actions::Add:
        PhysicsEngine::Log::info(("DIR (" + dir + ") FILE (" + filename + ") has event Add\n").c_str());
        break;
    case efsw::Actions::Delete:
        PhysicsEngine::Log::info(("DIR (" + dir + ") FILE (" + filename + ") has event Delete\n").c_str());
        break;
    case efsw::Actions::Modified:
        PhysicsEngine::Log::info(("DIR (" + dir + ") FILE (" + filename + ") has event Modified\n").c_str());
        break;
    case efsw::Actions::Moved:
        PhysicsEngine::Log::info(("DIR (" + dir + ") FILE (" + filename + ") has event Moved from (" + oldFilename + ")\n").c_str());
        break;
    default:
        PhysicsEngine::Log::info("This should never happen!\n");
    }

    // If file created or modified, add to buffer to load into world
    if (action == efsw::Action::Add || action == efsw::Action::Modified)
    {
        ProjectDatabase::fileAddedToProject(std::filesystem::path(dir) / filename);
    }

    if (action == efsw::Action::Delete)
    {
        ProjectDatabase::fileDeletedFromProject(std::filesystem::path(dir) / filename);
    }
}

std::filesystem::path ProjectDatabase::mDataPath = std::filesystem::path();
std::map<const std::filesystem::path, PhysicsEngine::Guid> ProjectDatabase::mFilePathToId = std::map<const std::filesystem::path, PhysicsEngine::Guid>();
std::map<const PhysicsEngine::Guid, std::filesystem::path> ProjectDatabase::mIdToFilePath = std::map<const PhysicsEngine::Guid, std::filesystem::path>();
std::vector<std::filesystem::path> ProjectDatabase::mAddBuffer = std::vector<std::filesystem::path>();
std::vector<std::filesystem::path> ProjectDatabase::mDeleteBuffer = std::vector<std::filesystem::path>();
DirectoryListener ProjectDatabase::mListener = DirectoryListener();
efsw::FileWatcher ProjectDatabase::mFileWatcher = efsw::FileWatcher();
efsw::WatchID ProjectDatabase::mWatchID = 0;

void ProjectDatabase::watch(const std::filesystem::path& projectPath)
{
    ProjectDatabase::mAddBuffer.clear();
    ProjectDatabase::mDeleteBuffer.clear();

    ProjectDatabase::mDataPath = projectPath / "data";

    // get all data files in project
    for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(ProjectDatabase::mDataPath))
    {
        if (std::filesystem::is_regular_file(entry))
        {
            ProjectDatabase::mAddBuffer.push_back(entry.path());
        }
    }

    // remove old watch
    ProjectDatabase::mFileWatcher.removeWatch(ProjectDatabase::mWatchID);

    // add watch for project data path to detect file changes
    ProjectDatabase::mWatchID = ProjectDatabase::mFileWatcher.addWatch(ProjectDatabase::mDataPath.string(), &ProjectDatabase::mListener, true);

    ProjectDatabase::mFileWatcher.watch();
}

void ProjectDatabase::update(PhysicsEngine::World* world)
{
    // load any assets queued up in add buffer into world
    for (size_t i = 0; i < ProjectDatabase::mAddBuffer.size(); i++)
    {
        std::string extension = ProjectDatabase::mAddBuffer[i].extension().string();

        PhysicsEngine::Asset* asset = nullptr;
        if (isAssetYamlExtension(extension))
        {
            asset = world->loadAssetFromYAML(ProjectDatabase::mAddBuffer[i].string());
        }

        // ensure each png file has a generated yaml texture file and if not then create one
        if (isTextureExtension(extension))
        {
            std::string texturePath = ProjectDatabase::mAddBuffer[i].string().substr(0, ProjectDatabase::mAddBuffer[i].string().find_last_of(".")) + ".texture";
            if (!std::filesystem::exists(texturePath))
            {
                PhysicsEngine::Texture2D* texture = world->createAsset<PhysicsEngine::Texture2D>();
                texture->load(ProjectDatabase::mAddBuffer[i].string());
                if (ProjectDatabase::mAddBuffer[i].has_stem())
                {
                    texture->setName(ProjectDatabase::mAddBuffer[i].stem().string());
                }
                else
                {
                    texture->setName(ProjectDatabase::mAddBuffer[i].filename().string());
                }
                texture->writeToYAML(texturePath);

                asset = texture;
            }
        }

        // ensure each obj file has a generated yaml mesh file and if not then create one
        if (isMeshExtension(extension))
        {
            std::string meshPath = ProjectDatabase::mAddBuffer[i].string().substr(0, ProjectDatabase::mAddBuffer[i].string().find_last_of(".")) + ".mesh";
            if (!std::filesystem::exists(meshPath))
            {
                PhysicsEngine::Mesh* mesh = world->createAsset<PhysicsEngine::Mesh>();
                mesh->load(ProjectDatabase::mAddBuffer[i].string());
                if (ProjectDatabase::mAddBuffer[i].has_stem())
                {
                    mesh->setName(ProjectDatabase::mAddBuffer[i].stem().string());
                }
                else
                {
                    mesh->setName(ProjectDatabase::mAddBuffer[i].filename().string());
                }
                mesh->writeToYAML(meshPath);

                asset = mesh;
            }
        }

        // ensure each glsl file has a generated yaml shader file and if not then create one
        if (isShaderExtension(extension))
        {
            std::string shaderPath = ProjectDatabase::mAddBuffer[i].string().substr(0, ProjectDatabase::mAddBuffer[i].string().find_last_of(".")) + ".shader";
            if (!std::filesystem::exists(shaderPath))
            {
                PhysicsEngine::Shader* shader = world->createAsset<PhysicsEngine::Shader>();

                PhysicsEngine::ShaderCreationAttrib attrib;
                attrib.mSourceFilepath = ProjectDatabase::mAddBuffer[i].string();
                attrib.mSourceLanguage = PhysicsEngine::ShaderSourceLanguage::GLSL;
                attrib.mVariantMacroMap[0] = { PhysicsEngine::ShaderMacro::None };

                if (ProjectDatabase::mAddBuffer[i].has_stem())
                {
                    attrib.mName = ProjectDatabase::mAddBuffer[i].stem().string();
                    shader->setName(ProjectDatabase::mAddBuffer[i].stem().string());
                }
                else
                {
                    attrib.mName = ProjectDatabase::mAddBuffer[i].filename().string();
                    shader->setName(ProjectDatabase::mAddBuffer[i].filename().string());
                }

                shader->load(attrib);
                shader->writeToYAML(shaderPath);

                asset = shader;
            }
        }

        if (asset != nullptr)
        {
            mFilePathToId[ProjectDatabase::mAddBuffer[i]] = asset->getGuid();
            mIdToFilePath[asset->getGuid()] = ProjectDatabase::mAddBuffer[i];
        }
        else
        {
            PhysicsEngine::Guid fileId = PhysicsEngine::Guid::newGuid();
            mFilePathToId[ProjectDatabase::mAddBuffer[i]] = fileId;
            mIdToFilePath[fileId] = ProjectDatabase::mAddBuffer[i];
        }
    }

    // clear buffer
    ProjectDatabase::mAddBuffer.clear();

    // destroy any assets queued up in delete buffer from world
    for (size_t i = 0; i < ProjectDatabase::mDeleteBuffer.size(); i++)
    {
        std::string extension = ProjectDatabase::mDeleteBuffer[i].extension().string();
        PhysicsEngine::Guid guid = getGuid(mDeleteBuffer[i]);

        if (isAssetYamlExtension(extension))
        {
            PhysicsEngine::Log::warn(("Deleting asset with id: " + guid.toString() + "\n").c_str());
            world->immediateDestroyAsset(guid, world->getTypeOf(guid));
        }

        mFilePathToId.erase(ProjectDatabase::mDeleteBuffer[i]);
        mIdToFilePath.erase(guid);
    }

    // clear buffer
    ProjectDatabase::mDeleteBuffer.clear();
}

void ProjectDatabase::fileAddedToProject(const std::filesystem::path& filePath)
{
    ProjectDatabase::mAddBuffer.push_back(filePath);
}

void ProjectDatabase::fileDeletedFromProject(const std::filesystem::path& filePath)
{
    ProjectDatabase::mDeleteBuffer.push_back(filePath);
}

void ProjectDatabase::createDirectory(const std::filesystem::path& parentPath)
{
    int i = 0;
    while (true)
    {
        std::string foldername = std::string("Folder (" + std::to_string(i++) + ")");
        std::filesystem::path folderPath = parentPath / foldername;
        if (std::filesystem::create_directory(folderPath))
        {
            break;
        }
    }
}

void ProjectDatabase::createShaderFile(const std::filesystem::path& parentPath)
{
    int i = 0;
    while (true)
    {
        std::string filename = ("Source(" + std::to_string(i++) + ").glsl");
        std::filesystem::path filepath = parentPath / filename;

        if (!std::filesystem::exists(filepath))
        {
            std::ofstream file(filepath);
            file << "#vertex\n";
            file << "#fragment\n";
            file.close();
            break;
        }
    }
}

void ProjectDatabase::createCubemapFile(PhysicsEngine::World* world, const std::filesystem::path& parentPath)
{
    int i = 0;
    while (true)
    {
        std::string filename = ("New Cubemap(" + std::to_string(i++) + ").cubemap");
        std::filesystem::path filepath = parentPath / filename;

        if (!std::filesystem::exists(filepath))
        {
            PhysicsEngine::Cubemap* cubemap = world->createAsset<PhysicsEngine::Cubemap>();
            cubemap->setName(filename);
            cubemap->writeToYAML(filepath.string());
            break;
        }
    }
}

void ProjectDatabase::createMaterialFile(PhysicsEngine::World* world, const std::filesystem::path& parentPath)
{
    int i = 0;
    while (true)
    {
        std::string filename = ("New Material(" + std::to_string(i++) + ").material");
        std::filesystem::path filepath = parentPath / filename;

        if (!std::filesystem::exists(filepath))
        {
            PhysicsEngine::Material* material = world->createAsset<PhysicsEngine::Material>();
            material->setName(filename);
            material->writeToYAML(filepath.string());
            break;
        }
    }
}

void ProjectDatabase::createSpriteFile(PhysicsEngine::World* world, const std::filesystem::path& parentPath)
{
    int i = 0;
    while (true)
    {
        std::string filename = ("New Sprite(" + std::to_string(i++) + ").sprite");
        std::filesystem::path filepath = parentPath / filename;

        if (!std::filesystem::exists(filepath))
        {
            PhysicsEngine::Sprite* sprite = world->createAsset<PhysicsEngine::Sprite>();
            sprite->writeToYAML(filepath.string());
            break;
        }
    }
}

void ProjectDatabase::createRenderTextureFile(PhysicsEngine::World* world, const std::filesystem::path& parentPath)
{
    int i = 0;
    while (true)
    {
        std::string filename = ("New RenderTexture(" + std::to_string(i++) + ").rendertexture");
        std::filesystem::path filepath = parentPath / filename;

        if (!std::filesystem::exists(filepath))
        {
            PhysicsEngine::RenderTexture* renderTexture = world->createAsset<PhysicsEngine::RenderTexture>();
            renderTexture->writeToYAML(filepath.string());
            break;
        }
    }
}

void ProjectDatabase::move(std::filesystem::path& oldPath, std::filesystem::path& newPath)
{
    std::error_code errorCode;
    if (std::filesystem::is_directory(oldPath, errorCode))
    {
        std::vector<std::filesystem::path> oldPaths;
        std::vector<std::filesystem::path> newPaths;

        for (auto& path : std::filesystem::recursive_directory_iterator(oldPath))
        {
            if (std::filesystem::is_regular_file(path, errorCode))
            {
                oldPaths.push_back(path);
                
                std::filesystem::path temp = std::filesystem::relative(path, oldPath);
             
                newPaths.push_back(newPath / temp);
            }
        }

        std::filesystem::rename(oldPath, newPath, errorCode);

        if (!errorCode)
        {
            for (size_t i = 0; i < oldPaths.size(); i++)
            {
                PhysicsEngine::Guid temp = getGuid(oldPaths[i]);

                mFilePathToId.erase(oldPaths[i]);
                mIdToFilePath.erase(temp);

                mFilePathToId[newPaths[i]] = temp;
                mIdToFilePath[temp] = newPaths[i];
            }
        }
    }
    else if (std::filesystem::is_regular_file(oldPath, errorCode))
    {
        std::filesystem::rename(oldPath, newPath, errorCode);

        if (!errorCode)
        {
            PhysicsEngine::Guid temp = getGuid(oldPath);

            mFilePathToId.erase(oldPath);
            mIdToFilePath.erase(temp);

            mFilePathToId[newPath] = temp;
            mIdToFilePath[temp] = newPath;
        }
    }
}

void ProjectDatabase::rename(std::filesystem::path& oldPath, std::string& newFilename)
{

}














PhysicsEngine::Guid ProjectDatabase::getGuid(const std::filesystem::path& filePath)
{
    std::map<const std::filesystem::path, PhysicsEngine::Guid>::const_iterator it = ProjectDatabase::mFilePathToId.find(filePath);
    if (it != ProjectDatabase::mFilePathToId.end())
    {
        return it->second;
    }

    return PhysicsEngine::Guid::INVALID;
}

std::filesystem::path ProjectDatabase::getFilePath(const PhysicsEngine::Guid& guid)
{
    std::map<const PhysicsEngine::Guid, std::filesystem::path>::const_iterator it = ProjectDatabase::mIdToFilePath.find(guid);
    if (it != ProjectDatabase::mIdToFilePath.end())
    {
        return it->second;
    }

    return std::filesystem::path();
}

bool ProjectDatabase::isAssetYamlExtension(const std::string& extension)
{
    if (extension == ".texture" ||
        extension == ".mesh" ||
        extension == ".shader" ||
        extension == ".material" ||
        extension == ".sprite" ||
        extension == ".rendertexture" ||
        extension == ".cubemap")
    {
        return true;
    }

    return false;
}

bool ProjectDatabase::isTextureExtension(const std::string& extension)
{
    if (extension == ".png" ||
        extension == ".jpg")
    {
        return true;
    }

    return false;
}

bool ProjectDatabase::isMeshExtension(const std::string& extension)
{
    if (extension == ".obj")
    {
        return true;
    }

    return false;
}

bool ProjectDatabase::isShaderExtension(const std::string& extension)
{
    if (extension == ".glsl" ||
        extension == ".hlsl")
    {
        return true;
    }

    return false;
}