#include "../include/ProjectDatabase.h"

#include <core/Util.h>

#include <fstream>
#include <random>

using namespace PhysicsEditor;

void DirectoryListener::handleFileAction(efsw::WatchID watchid, const std::string& dir, const std::string& filename, efsw::Action action, std::string oldFilename) 
{
    std::filesystem::path path = std::filesystem::path(dir) / filename;

    Action a;
    a.mPath = path;
    a.mAction = action;

    ProjectDatabase::queueFileAction(a);

    switch (a.mAction) {
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
    }
}

std::filesystem::path ProjectDatabase::mDataPath = std::filesystem::path();
std::map<const std::filesystem::path, PhysicsEngine::Guid> ProjectDatabase::mFilePathToId = std::map<const std::filesystem::path, PhysicsEngine::Guid>();
std::map<const PhysicsEngine::Guid, std::filesystem::path> ProjectDatabase::mIdToFilePath = std::map<const PhysicsEngine::Guid, std::filesystem::path>();
std::queue<Action> ProjectDatabase::mActionQueue = std::queue<Action>();
DirectoryListener ProjectDatabase::mListener = DirectoryListener();
efsw::FileWatcher ProjectDatabase::mFileWatcher = efsw::FileWatcher();
efsw::WatchID ProjectDatabase::mWatchID = 0;

void ProjectDatabase::watch(const std::filesystem::path& projectPath)
{
    ProjectDatabase::mActionQueue = {};

    ProjectDatabase::mDataPath = projectPath / "data";

    // get all data files in project
    for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(ProjectDatabase::mDataPath))
    {
        if (std::filesystem::is_regular_file(entry))
        {
            Action action;
            action.mPath = entry.path();
            action.mAction = efsw::Action::Add;
            ProjectDatabase::mActionQueue.push(action);
        }
    }

    // remove old watch
    ProjectDatabase::mFileWatcher.removeWatch(ProjectDatabase::mWatchID);

    // add watch for project data path to detect file changes
    ProjectDatabase::mWatchID = ProjectDatabase::mFileWatcher.addWatch(ProjectDatabase::mDataPath.string(), &ProjectDatabase::mListener, true);

    ProjectDatabase::mFileWatcher.watch();
}

void ProjectDatabase::addFile(const std::filesystem::path& path, PhysicsEngine::World* world)
{
    std::string extension = path.extension().string();

    PhysicsEngine::Asset* asset = nullptr;
    if (PhysicsEngine::Util::isAssetYamlExtension(extension))
    {
        asset = world->loadAssetFromYAML(path.string());
        PhysicsEngine::Log::warn(("Loading asset with id: " + asset->getGuid().toString() + "\n").c_str());
    }

    // ensure each png file has a generated yaml texture file and if not then create one
    if (PhysicsEngine::Util::isTextureExtension(extension))
    {
        std::string texturePath = path.string().substr(0, path.string().find_last_of(".")) + PhysicsEngine::TEXTURE2D_EXT;
        if (!std::filesystem::exists(texturePath))
        {
            PhysicsEngine::Texture2D* texture = world->createAsset<PhysicsEngine::Texture2D>();
            texture->load(path.string());
            if (path.has_stem())
            {
                texture->setName(path.stem().string());
            }
            else
            {
                texture->setName(path.filename().string());
            }
            texture->writeToYAML(texturePath);

            asset = texture;
        }
    }

    // ensure each obj file has a generated yaml mesh file and if not then create one
    if (PhysicsEngine::Util::isMeshExtension(extension))
    {
        std::string meshPath = path.string().substr(0, path.string().find_last_of(".")) + PhysicsEngine::MESH_EXT;
        if (!std::filesystem::exists(meshPath))
        {
            PhysicsEngine::Mesh* mesh = world->createAsset<PhysicsEngine::Mesh>();
            mesh->load(path.string());
            if (path.has_stem())
            {
                mesh->setName(path.stem().string());
            }
            else
            {
                mesh->setName(path.filename().string());
            }
            mesh->writeToYAML(meshPath);

            asset = mesh;
        }
    }

    // ensure each glsl file has a generated yaml shader file and if not then create one
    if (PhysicsEngine::Util::isShaderExtension(extension))
    {
        std::string shaderPath = path.string().substr(0, path.string().find_last_of(".")) + PhysicsEngine::SHADER_EXT;
        if (!std::filesystem::exists(shaderPath))
        {
            PhysicsEngine::Shader* shader = world->createAsset<PhysicsEngine::Shader>();

            PhysicsEngine::ShaderCreationAttrib attrib;
            attrib.mSourceFilepath = path.string();
            attrib.mSourceLanguage = PhysicsEngine::ShaderSourceLanguage::GLSL;
            attrib.mVariantMacroMap[0] = { PhysicsEngine::ShaderMacro::None };

            if (path.has_stem())
            {
                attrib.mName = path.stem().string();
                shader->setName(path.stem().string());
            }
            else
            {
                attrib.mName = path.filename().string();
                shader->setName(path.filename().string());
            }

            shader->load(attrib);
            shader->writeToYAML(shaderPath);

            asset = shader;
        }
    }

    if (asset != nullptr)
    {
        mFilePathToId[path] = asset->getGuid();
        mIdToFilePath[asset->getGuid()] = path;
    }
    else
    {
        PhysicsEngine::Guid fileId = PhysicsEngine::Guid::newGuid();
        mFilePathToId[path] = fileId;
        mIdToFilePath[fileId] = path;
    }
}

void ProjectDatabase::deleteFile(const std::filesystem::path& path, PhysicsEngine::World* world)
{
    std::string extension = path.extension().string();
    PhysicsEngine::Guid guid = getGuid(path);

    if (PhysicsEngine::Util::isAssetYamlExtension(extension))
    {
        PhysicsEngine::Log::warn(("Deleting asset with id: " + guid.toString() + "\n").c_str());
        world->immediateDestroyAsset(guid, world->getTypeOf(guid));
    }

    mFilePathToId.erase(path);
    mIdToFilePath.erase(guid);
}

void ProjectDatabase::update(PhysicsEngine::World* world)
{
    while (!ProjectDatabase::mActionQueue.empty())
    {
        Action action = ProjectDatabase::mActionQueue.front();
        ProjectDatabase::mActionQueue.pop();

        switch (action.mAction)
        {
        case efsw::Actions::Add:
        case efsw::Actions::Modified:
        case efsw::Actions::Moved:
            ProjectDatabase::addFile(action.mPath, world);
            break;
        case efsw::Actions::Delete:
            ProjectDatabase::deleteFile(action.mPath, world);
            break;
        }
    }
}
    
void ProjectDatabase::queueFileAction(Action action)
{
    ProjectDatabase::mActionQueue.push(action);
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

void ProjectDatabase::rename(const std::filesystem::path& oldPath, const std::filesystem::path& newPath)
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
    }
}

void ProjectDatabase::remove_all(const std::filesystem::path& path)
{

}

void ProjectDatabase::remove(const std::filesystem::path& path)
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
















void ProjectDatabase::newProject(Clipboard& clipboard, const std::filesystem::path& projectPath)
{
    std::string message = "newProject project name: " + projectPath.filename().string() + " project path: " + projectPath.string() + "\n";
    PhysicsEngine::Log::info(message.c_str());

    if (projectPath.empty())
    {
        return;
    }

    if (std::filesystem::create_directory(projectPath))
    {
        bool success = true;
        success &= std::filesystem::create_directory(projectPath / "data");
        success &= std::filesystem::create_directory(projectPath / "data/scenes");
        success &= std::filesystem::create_directory(projectPath / "data/textures");
        success &= std::filesystem::create_directory(projectPath / "data/meshes");
        success &= std::filesystem::create_directory(projectPath / "data/materials");
        success &= std::filesystem::create_directory(projectPath / "data/shaders");
        success &= std::filesystem::create_directory(projectPath / "data/sprites");

        if (!success)
        {
            PhysicsEngine::Log::error("Could not create project sub directories\n");
            return;
        }
    }
    else
    {
        PhysicsEngine::Log::error("Could not create project root directory\n");
        return;
    }

    // mark any (non-editor) entities in currently opened scene to be immediately destroyed
    clipboard.getWorld()->getActiveScene()->immediateDestroyEntitiesInScene();

    // tell library directory which project to watch
    ProjectDatabase::watch(projectPath.string());

    // reset editor camera
    clipboard.mCameraSystem->resetCamera();

    clipboard.setActiveProject(projectPath.filename().string(), projectPath.string());
    clipboard.setActiveScene("", "", PhysicsEngine::Guid::INVALID);
}

void ProjectDatabase::openProject(Clipboard& clipboard, const std::filesystem::path& projectPath)
{
    std::string message = "newProject project name: " + projectPath.filename().string() + " project path: " + projectPath.string() + "\n";
    PhysicsEngine::Log::info(message.c_str());

    if (projectPath.empty())
    {
        return;
    }

    // mark any (non-editor) entities in currently opened scene to be immediately destroyed
    clipboard.getWorld()->getActiveScene()->immediateDestroyEntitiesInScene();

    // tell library directory which project to watch
    ProjectDatabase::watch(projectPath);

    // reset editor camera
    clipboard.mCameraSystem->resetCamera();

    clipboard.setActiveProject(projectPath.filename().string(), projectPath.string());
    clipboard.setActiveScene("", "", PhysicsEngine::Guid::INVALID);
}

void ProjectDatabase::saveProject(Clipboard& clipboard)
{
    for (auto it = clipboard.mModifiedAssets.begin(); it != clipboard.mModifiedAssets.end(); it++)
    {
        std::string path = ProjectDatabase::getFilePath(*it).string();
        if (!path.empty()) {
            clipboard.getWorld()->writeAssetToYAML(path, *it);
        }
    }

    clipboard.mModifiedAssets.clear();
}

void ProjectDatabase::newScene(Clipboard& clipboard, const std::string& sceneName)
{
    std::string message = "newScene scene name: " + sceneName + "\n";
    PhysicsEngine::Log::info(message.c_str());

    // check that we have an open project
    if (clipboard.getProjectPath().empty())
    {
        return;
    }

    // mark any (non-editor) entities in currently opened scene to be immediately destroyed
    clipboard.getWorld()->getActiveScene()->immediateDestroyEntitiesInScene();

    // re-centre editor camera to default position
    clipboard.mCameraSystem->resetCamera();

    // clear any dragged and selected items on clipboard
    clipboard.clearDraggedItem();
    clipboard.clearSelectedItem();

    PhysicsEngine::Scene* scene = clipboard.getWorld()->createScene();
    if (scene != nullptr)
    {
        clipboard.setActiveScene(sceneName, "", scene->getGuid());
    }
}

void ProjectDatabase::openScene(Clipboard& clipboard, const std::filesystem::path& scenePath)
{
    std::string message = "openScene scene name: " + scenePath.filename().string() + " scene path: " + scenePath.string() + "\n";
    PhysicsEngine::Log::info(message.c_str());

    // check that we have an open project
    if (clipboard.getProjectPath().empty())
    {
        return;
    }

    // check to make sure the scene is part of the current project
    if (scenePath.string().find(clipboard.getProjectPath().string()) != 0)
    {
        return;
    }

    // mark any (non-editor) entities in currently opened scene to be immediately destroyed
    clipboard.getWorld()->getActiveScene()->immediateDestroyEntitiesInScene();

    // reset editor camera to default position
    clipboard.mCameraSystem->resetCamera();

    // clear any dragged and selected items on clipboard
    clipboard.clearDraggedItem();
    clipboard.clearSelectedItem();

    // load scene into world
    PhysicsEngine::Scene* scene = clipboard.getWorld()->loadSceneFromYAML(scenePath.string());
    if (scene != nullptr)
    {
        clipboard.setActiveScene(scenePath.filename().string(), scenePath.string(), scene->getGuid());
    }
}

void ProjectDatabase::saveScene(Clipboard& clipboard, const std::filesystem::path& scenePath)
{
    std::string message = "saveScene scene name: " + scenePath.filename().string() + " scene path: " + scenePath.string() + "\n";
    PhysicsEngine::Log::info(message.c_str());

    if (scenePath.empty())
    {
        return;
    }

    clipboard.setActiveScene(scenePath.filename().string(), scenePath.string(), clipboard.getSceneId());

    clipboard.getWorld()->writeSceneToYAML(scenePath.string(), clipboard.getSceneId());
}

void ProjectDatabase::populateScene(Clipboard& clipboard)
{
    int m = 20;
    int n = 20;
    std::vector<int> layout(m * n, 0);

    std::random_device                  rand_dev;
    std::mt19937                        generator(rand_dev());
    std::uniform_int_distribution<int>  distr(0, 10);

    for (size_t i = 0; i < layout.size(); i++)
    {
        layout[i] = distr(generator);
    }

    PhysicsEngine::Entity* lightEntity = clipboard.getWorld()->getActiveScene()->createLight(PhysicsEngine::LightType::Directional);
    lightEntity->setName("Light");

    PhysicsEngine::Entity* planeEntity = clipboard.getWorld()->getActiveScene()->createPrimitive(PhysicsEngine::PrimitiveType::Plane);
    planeEntity->setName("Plane");
    PhysicsEngine::Transform* planeTransform = planeEntity->getComponent<PhysicsEngine::Transform>();
    planeTransform->setPosition(glm::vec3(0, 0, 0));
    planeTransform->setScale(glm::vec3(50, 1, 50));

    int index = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int l = layout[n * i + j];
            for (int k = 0; k < l; k++)
            {
                std::string name = "Cube" + std::to_string(index++);

                PhysicsEngine::Entity* entity = clipboard.getWorld()->getActiveScene()->createPrimitive(PhysicsEngine::PrimitiveType::Cube);
                entity->setName(name);
                PhysicsEngine::Transform* transform = entity->getComponent<PhysicsEngine::Transform>();
                transform->setPosition(glm::vec3(i + 0.5f, k + 0.5f, j + 0.5f));
                //transform->mPosition = glm::vec3(i + 0.5f, k + 0.5f, j + 0.5f);
            }
        }
    }
}