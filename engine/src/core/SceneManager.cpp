#include "../../include/core/SceneManager.h"

using namespace PhysicsEngine;

SceneManager::SceneManager()
{
	manager = new Manager();
}

SceneManager::~SceneManager()
{
	delete manager;
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

	if(!manager->validate(scenes, assets)){
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

		manager->load(*loadingScene, assets);

		for(int i = 0; i < manager->getNumberOfSystems(); i++){
			System* system = manager->getSystemByIndex(i);

			system->setSceneContext(&context);
			system->init();
		}

		activeSceneIndex = loadingSceneIndex;
		activeScene = loadingScene;
		loadingScene = NULL;
	}

	if(activeScene != NULL){
		for(int i = 0; i < manager->getNumberOfSystems(); i++){
			System* system = manager->getSystemByIndex(i);

			system->update();
		}
	}
	
}