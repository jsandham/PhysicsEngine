#include "../../include/core/WorldManager.h"
#include "../../include/core/Log.h"

using namespace PhysicsEngine;

WorldManager::WorldManager()
{
}

WorldManager::~WorldManager()
{
}

bool WorldManager::load(std::string sceneFilePath, std::vector<std::string> assetFilePaths)
{
    for (size_t i = 0; i < assetFilePaths.size(); i++)
    {
        if (!world.loadAsset(assetFilePaths[i]))
        {
            Log::error("Could not load asset file\n");
            return false;
        }
    }

    if (!world.loadScene(sceneFilePath))
    {
        Log::error("Could not load scene file\n");
        return false;
    }

    return true;
}

void WorldManager::init()
{
    for (size_t i = 0; i < world.getNumberOfUpdatingSystems(); i++)
    {
        System *system = world.getSystemByUpdateOrder(i);

        system->init(&world);
    }
}

void WorldManager::update(Time time, Input input)
{
    for (size_t i = 0; i < world.getNumberOfUpdatingSystems(); i++)
    {
        System *system = world.getSystemByUpdateOrder(i);

        system->update(input, time);
    }
}