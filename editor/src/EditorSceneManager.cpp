#include "../include/EditorSceneManager.h"

#include "core/World.h"
#include "core/Log.h"

using namespace PhysicsEditor;

void EditorSceneManager::newScene(Clipboard& clipboard, const std::string& sceneName)
{
    std::string message = "newScene scene name: " + sceneName + "\n";
    PhysicsEngine::Log::info(message.c_str());

    // check that we have an open project
    if (clipboard.getProjectPath().empty())
    {
        return;
    }

    // mark any (non-editor) entities in currently opened scene to be immediately destroyed
    clipboard.getWorld()->immediateDestroyEntitiesInWorld();

    // re-centre editor camera to default position
    clipboard.mCameraSystem->resetCamera();

    // clear any dragged and selected items on clipboard
    clipboard.clearDraggedItem();
    clipboard.clearSelectedItem();

    PhysicsEngine::Scene* scene = clipboard.getWorld()->createScene();
    if (scene != nullptr)
    {
        clipboard.setActiveScene(sceneName, "", scene->getId());
    }
}

void EditorSceneManager::openScene(Clipboard& clipboard, const std::filesystem::path& scenePath)
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
    clipboard.getWorld()->immediateDestroyEntitiesInWorld();

    // reset editor camera to default position
    clipboard.mCameraSystem->resetCamera();

    // clear any dragged and selected items on clipboard
    clipboard.clearDraggedItem();
    clipboard.clearSelectedItem();

    // load scene into world
    PhysicsEngine::Scene* scene = clipboard.getWorld()->loadSceneFromYAML(scenePath.string());
    if (scene != nullptr)
    {
        clipboard.setActiveScene(scenePath.filename().string(), scenePath.string(), scene->getId());
    }
}

void EditorSceneManager::saveScene(Clipboard& clipboard, const std::filesystem::path& scenePath)
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