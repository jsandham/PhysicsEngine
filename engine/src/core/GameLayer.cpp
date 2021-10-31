#include "../../include/core/Log.h"
#include "../../include/core/Input.h"
#include "../../include/core/GameLayer.h"

#include <filesystem>
#include <stack>

using namespace PhysicsEngine;
namespace fs = std::filesystem;

GameLayer::GameLayer() : Layer("Game Layer")
{
}

GameLayer::~GameLayer()
{
}

void GameLayer::init()
{
    fs::path cwd = fs::current_path();
    fs::path dataPath[2] = { cwd / "data", cwd / "game_data"};

    for (int i = 0; i < 2; i++)
    {
        std::cout << "cwd: " << cwd.string() << std::endl;
        std::cout << "dataPath: " << dataPath[i].string() << std::endl;

        if (fs::is_directory(dataPath[i]))
        {
            std::stack<fs::path> stack;
            stack.push(dataPath[i]);

            while (!stack.empty())
            {
                fs::path currentPath = stack.top();
                stack.pop();

                std::error_code error_code;
                for (const fs::directory_entry& entry : fs::directory_iterator(currentPath, error_code))
                {
                    if (fs::is_directory(entry, error_code))
                    {
                        stack.push(entry.path());
                    }
                    else if (fs::is_regular_file(entry, error_code))
                    {
                        std::string extension = entry.path().extension().string();
                        if (extension == ".mesh" ||
                            extension == ".shader" ||
                            extension == ".material" ||
                            extension == ".texture")
                        {
                            fs::path relativeDataPath = entry.path().lexically_relative(fs::current_path());
                            std::cout << "relativeDataPath: " << relativeDataPath.string() << std::endl;
                            mWorld.loadAssetFromYAML(relativeDataPath.string());
                        }
                    }
                }
            }
        }
    }

    std::cout << "mesh count: " << mWorld.getNumberOfAssets<Mesh>() << std::endl;
    std::cout << "shader count: " << mWorld.getNumberOfAssets<Shader>() << std::endl;
    std::cout << "material count: " << mWorld.getNumberOfAssets<Material>() << std::endl;
    std::cout << "texture count: " << mWorld.getNumberOfAssets<Texture2D>() << std::endl;

    fs::path scenePath = cwd / "game_data\\scenes\\suzanne2.scene";
    std::cout << "scenePath: " << scenePath.string() << std::endl;

    mWorld.loadSceneFromYAML(scenePath.string());

    FreeLookCameraSystem* cameraSystem = mWorld.addSystem<FreeLookCameraSystem>(0);
    RenderSystem* renderSystem = mWorld.addSystem<RenderSystem>(1);
    CleanUpSystem* cleanupSystem = mWorld.addSystem<CleanUpSystem>(2);

    std::cout << "camerasystem count: " << mWorld.getNumberOfSystems<FreeLookCameraSystem>() << std::endl;
    std::cout << "rendersystem count: " << mWorld.getNumberOfSystems<RenderSystem>() << std::endl;
    std::cout << "cleanupsystem count: " << mWorld.getNumberOfSystems<CleanUpSystem>() << std::endl;

    for (size_t i = 0; i < mWorld.getNumberOfUpdatingSystems(); i++)
    {
        System* system = mWorld.getSystemByUpdateOrder(i);

        system->init(&mWorld);
    }

    CameraSettings settings;
    settings.mRenderToScreen = true;

    cameraSystem->configureCamera(settings);
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