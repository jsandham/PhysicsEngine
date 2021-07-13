#include "../include/EditorProjectManager.h"

#include "core/World.h"

using namespace PhysicsEditor;

void EditorProjectManager::newProject(Clipboard& clipboard)
{
}

void EditorProjectManager::openProject(Clipboard& clipboard)
{
}

void EditorProjectManager::saveProject(Clipboard& clipboard)
{
    // save materials
    for (size_t i = 0; i < clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Material>(); i++)
    {
        PhysicsEngine::Material* material = clipboard.getWorld()->getAssetByIndex<PhysicsEngine::Material>(i);
        std::string path = clipboard.getWorld()->getAssetFilepath(material->getId());
        if (!path.empty()) {
            clipboard.getWorld()->writeAssetToYAML(path, material->getId());
        }
    }

    // save sprites
    for (size_t i = 0; i < clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Sprite>(); i++)
    {
        PhysicsEngine::Sprite* sprite = clipboard.getWorld()->getAssetByIndex<PhysicsEngine::Sprite>(i);
        std::string path = clipboard.getWorld()->getAssetFilepath(sprite->getId());
        if (!path.empty()) {
            clipboard.getWorld()->writeAssetToYAML(path, sprite->getId());
        }
    }
}