#include "../include/EditorSceneManager.h"

#include "core/World.h"
#include "core/Log.h"

#include <random>

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

void EditorSceneManager::populateScene(Clipboard& clipboard)
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

    PhysicsEngine::Entity* lightEntity = clipboard.getWorld()->createLight(PhysicsEngine::LightType::Directional);
    lightEntity->setName("Light");

    PhysicsEngine::Entity* planeEntity = clipboard.getWorld()->createPrimitive(PhysicsEngine::PrimitiveType::Plane);
    planeEntity->setName("Plane");
    PhysicsEngine::Transform* planeTransform = planeEntity->getComponent<PhysicsEngine::Transform>();
    planeTransform->mPosition = glm::vec3(0, 0, 0);
    planeTransform->mScale = glm::vec3(50, 1, 50);

    int index = 0;
    for (int i = 0; i < m; i++) 
    {
        for (int j = 0; j < n; j++)
        {
            int l = layout[n * i + j];
            for (int k = 0; k < l; k++)
            {
                std::string name = "Cube" + std::to_string(index++);

                PhysicsEngine::Entity* entity = clipboard.getWorld()->createPrimitive(PhysicsEngine::PrimitiveType::Cube);
                entity->setName(name);
                PhysicsEngine::Transform* transform = entity->getComponent<PhysicsEngine::Transform>();
                transform->mPosition = glm::vec3(i + 0.5f, k + 0.5f, j + 0.5f);
            }
        }
    }
}