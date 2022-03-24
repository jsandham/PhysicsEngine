#include "../include/EditorProjectManager.h"

#include "core/World.h"
#include "core/Log.h"

using namespace PhysicsEditor;

void EditorProjectManager::newProject(Clipboard& clipboard, const std::filesystem::path& projectPath)
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
    clipboard.getWorld()->immediateDestroyEntitiesInWorld();

    // tell library directory which project to watch
    clipboard.getLibrary().watch(projectPath.string());

    // reset editor camera
    clipboard.mCameraSystem->resetCamera();

    clipboard.setActiveProject(projectPath.filename().string(), projectPath.string());
    clipboard.setActiveScene("", "", PhysicsEngine::Guid::INVALID);
}

void EditorProjectManager::openProject(Clipboard& clipboard, const std::filesystem::path& projectPath)
{
    std::string message = "newProject project name: " + projectPath.filename().string() + " project path: " + projectPath.string() + "\n";
    PhysicsEngine::Log::info(message.c_str());

    if (projectPath.empty())
    {
        return;
    }

    // mark any (non-editor) entities in currently opened scene to be immediately destroyed
    clipboard.getWorld()->immediateDestroyEntitiesInWorld();

    // tell library directory which project to watch
    clipboard.getLibrary().watch(projectPath);

    // reset editor camera
    clipboard.mCameraSystem->resetCamera();

    clipboard.setActiveProject(projectPath.filename().string(), projectPath.string());
    clipboard.setActiveScene("", "", PhysicsEngine::Guid::INVALID);
}

void EditorProjectManager::saveProject(Clipboard& clipboard)
{
    for (auto it = clipboard.mModifiedAssets.begin(); it != clipboard.mModifiedAssets.end(); it++)
    {
        std::string path = clipboard.getWorld()->getAssetFilepath(*it);
        if (!path.empty()) {
            clipboard.getWorld()->writeAssetToYAML(path, *it);
        }
    }

    clipboard.mModifiedAssets.clear();
}