#include "../include/EditorSceneManager.h"
#include "../include/EditorCameraSystem.h"

#include "core/World.h"

using namespace PhysicsEditor;

void EditorSceneManager::newScene(Clipboard& clipboard)
{
    // mark any (non-editor) entities in currently opened scene to be latent destroyed
    //clipboard.getWorld()->latentDestroyEntitiesInWorld();
    clipboard.getWorld()->immediateDestroyEntitiesInWorld();

    // re-centre editor camera to default position
    clipboard.getWorld()->getSystem<PhysicsEngine::EditorCameraSystem>()->resetCamera();

    // clear any dragged and selected items on clipboard
    clipboard.clearDraggedItem();
    clipboard.clearSelectedItem();

    PhysicsEngine::Scene* scene = clipboard.getWorld()->createScene();
    if (scene != nullptr)
    {
        clipboard.setActiveScene("default.scene", "", scene->getId());
    }
}

void EditorSceneManager::openScene(Clipboard& clipboard, const std::string& name, const std::string& path)
{
    // check to make sure the scene is part of the current project
    if (path.find(clipboard.getProjectPath() + "\\data\\") != 0)
    {
        return;
    }

    // mark any (non-editor) entities in currently opened scene to be latent destroyed
    clipboard.getWorld()->immediateDestroyEntitiesInWorld();

    // reset editor camera to default position
    clipboard.getWorld()->getSystem<PhysicsEngine::EditorCameraSystem>()->resetCamera();

    // clear any dragged and selected items on clipboard
    clipboard.clearDraggedItem();
    clipboard.clearSelectedItem();

    // load scene into world
    PhysicsEngine::Scene* scene = clipboard.getWorld()->loadSceneFromYAML(path);
    if (scene != nullptr)
    {
        clipboard.setActiveScene(name, path, scene->getId());
    }
}

void EditorSceneManager::saveScene(Clipboard& clipboard, const std::string& name, const std::string& path)
{
    clipboard.getWorld()->writeSceneToYAML(path, clipboard.getSceneId());
}