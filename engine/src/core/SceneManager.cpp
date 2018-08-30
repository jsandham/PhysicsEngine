#include "../../include/core/SceneManager.h"

using namespace PhysicsEngine;

SceneManager::SceneManager()
{
	manager = new Manager();

	playerSystem = new PlayerSystem(manager);
	physicsSystem = new PhysicsSystem(manager);
	renderSystem = new RenderSystem(manager);
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

void SceneManager::init()
{
	if(scenes.size() == 0){
		std::cout << "Warning: No scenes found" << std::endl;
		return;
	}

	if(!validate(scenes, assets)){
		std::cout << "Error: Validation failed" << std::endl;
		return;
	}

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
		//playerSystem->update();
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