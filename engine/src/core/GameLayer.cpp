#include "../../include/core/GameLayer.h"
#include "../../include/core/Log.h"

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

        mWorld.loadAllAssetsInPath(dataPath[i]);
    }

    std::cout << "mesh count: " << mWorld.getNumberOfAssets<Mesh>() << std::endl;
    std::cout << "shader count: " << mWorld.getNumberOfAssets<Shader>() << std::endl;
    std::cout << "material count: " << mWorld.getNumberOfAssets<Material>() << std::endl;
    std::cout << "texture count: " << mWorld.getNumberOfAssets<Texture2D>() << std::endl;

    /*std::filesystem::path scenePath = cwd / "game_data\\scenes\\simple.scene";*/
    std::filesystem::path scenePath = cwd / "game_data\\scenes\\Terrain.scene";
    std::cout << "scenePath: " << scenePath.string() << std::endl;

    mWorld.loadSceneFromYAML(scenePath.string());

    FreeLookCameraSystem *cameraSystem = mWorld.getSystem<FreeLookCameraSystem>();
    TerrainSystem *terrainSystem = mWorld.getSystem<TerrainSystem>();
    RenderSystem *renderSystem = mWorld.getSystem<RenderSystem>();
    CleanUpSystem *cleanUpSystem = mWorld.getSystem<CleanUpSystem>();

    CameraSystemConfig config;
    config.mRenderToScreen = true;
    config.mSpawnCameraOnInit = false;
    cameraSystem->configureCamera(config);

    cameraSystem->init(&mWorld);
    terrainSystem->init(&mWorld);
    renderSystem->init(&mWorld);
    cleanUpSystem->init(&mWorld);
}

void GameLayer::begin()
{
}

void GameLayer::update()
{
    FreeLookCameraSystem *cameraSystem = mWorld.getSystem<FreeLookCameraSystem>();
    TerrainSystem *terrainSystem = mWorld.getSystem<TerrainSystem>();
    RenderSystem *renderSystem = mWorld.getSystem<RenderSystem>();
    CleanUpSystem *cleanUpSystem = mWorld.getSystem<CleanUpSystem>();

    cameraSystem->update();
    terrainSystem->update();
    renderSystem->update();
    cleanUpSystem->update();
}

void GameLayer::end()
{
}

bool GameLayer::quit()
{
    return false;
}