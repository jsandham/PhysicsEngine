#include "../../include/core/Log.h"
#include "../../include/core/Input.h"
#include "../../include/core/GameLayer.h"

#include <filesystem>
#include <stack>

using namespace PhysicsEngine;

GameLayer::GameLayer() : Layer("Game Layer")
{
}

GameLayer::~GameLayer()
{
}

void GameLayer::init()
{
    std::filesystem::path cwd = std::filesystem::current_path();
    std::filesystem::path dataPath[2] = {cwd / "data", cwd / "game_data"};

    for (int i = 0; i < 2; i++)
    {
        std::cout << "cwd: " << cwd.string() << std::endl;
        std::cout << "dataPath: " << dataPath[i].string() << std::endl;

        mWorld.loadAssetsInPath(dataPath[i]);
    }

    std::cout << "mesh count: " << mWorld.getNumberOfAssets<Mesh>() << std::endl;
    std::cout << "shader count: " << mWorld.getNumberOfAssets<Shader>() << std::endl;
    std::cout << "material count: " << mWorld.getNumberOfAssets<Material>() << std::endl;
    std::cout << "texture count: " << mWorld.getNumberOfAssets<Texture2D>() << std::endl;

    /*std::filesystem::path scenePath = cwd / "game_data\\scenes\\simple.scene";*/
    std::filesystem::path scenePath = cwd / "game_data\\scenes\\Terrain.scene";
    std::cout << "scenePath: " << scenePath.string() << std::endl;

    mWorld.loadSceneFromYAML(scenePath.string());

    FreeLookCameraSystem* cameraSystem = mWorld.addSystem<FreeLookCameraSystem>(0);
    mWorld.addSystem<TerrainSystem>(1);
    mWorld.addSystem<RenderSystem>(2);
    mWorld.addSystem<CleanUpSystem>(3);

    std::cout << "camerasystem count: " << mWorld.getNumberOfSystems<FreeLookCameraSystem>() << std::endl;
    std::cout << "terrainsystem count: " << mWorld.getNumberOfSystems<TerrainSystem>() << std::endl;
    std::cout << "rendersystem count: " << mWorld.getNumberOfSystems<RenderSystem>() << std::endl;
    std::cout << "cleanupsystem count: " << mWorld.getNumberOfSystems<CleanUpSystem>() << std::endl;

    for (size_t i = 0; i < mWorld.getNumberOfUpdatingSystems(); i++)
    {
        System* system = mWorld.getSystemByUpdateOrder(i);

        system->init(&mWorld);
    }

    cameraSystem->configureCamera();
}

void GameLayer::begin()
{

}

void GameLayer::update(const Time& time)
{
    for (size_t i = 0; i < mWorld.getNumberOfUpdatingSystems(); i++)
    {
        System* system = mWorld.getSystemByUpdateOrder(i);

        system->update(getInput(), time);
    }
}

void GameLayer::end()
{

}