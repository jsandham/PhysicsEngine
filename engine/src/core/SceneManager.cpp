#include "../../include/core/SceneManager.h"

using namespace PhysicsEngine;

SceneManager::SceneManager()
{
	manager = new Manager();

	playerSystem = new PlayerSystem(manager, &context);
	physicsSystem = new PhysicsSystem(manager, &context);
	renderSystem = new RenderSystem(manager, &context);
}

SceneManager::~SceneManager()
{
	delete manager;

	delete playerSystem;
	delete physicsSystem;
	delete renderSystem;
}

void SceneManager::add(Scene scene)
{
	scenes.push_back(scene);

	context.add(scene);
}

void SceneManager::add(Asset asset)
{
	assets.push_back(asset);
}

bool SceneManager::validate()
{
	if(scenes.size() == 0){
		std::cout << "Warning: No scenes found" << std::endl;
		return false;
	}

	if(!validate(scenes, assets)){
		std::cout << "Error: Validation failed" << std::endl;
		return false;
	}

	return true;
}

void SceneManager::init()
{
	loadingSceneIndex = -1;
	activeSceneIndex = -1;
}

void SceneManager::update()
{
	int sceneToLoadIndex = context.getSceneToLoadIndex();
	if(sceneToLoadIndex != activeSceneIndex){
		loadingSceneIndex = sceneToLoadIndex;
		loadingScene = &scenes[sceneToLoadIndex];
	}

	if(loadingScene != NULL){
		std::cout << "loading scene: " << loadingScene->filepath << std::endl;

		load(*loadingScene, assets);

		//compress();

		playerSystem->init(); 
		physicsSystem->init();
		renderSystem->init();

		activeSceneIndex = loadingSceneIndex;
		activeScene = loadingScene;
		loadingScene = NULL;
	}

	if(activeScene != NULL){
		//physicsSystem->update();
		//renderSystem->update();
		playerSystem->update();
	}
	
}

bool SceneManager::validate(std::vector<Scene> scenes, std::vector<Asset> assets)
{
	return manager->validate(scenes, assets);
}

void SceneManager::load(Scene scene, std::vector<Asset> assets)
{
	manager->load(scene, assets);
}

// void SceneManager::compress()
// {
// 	manager->compress();
// }