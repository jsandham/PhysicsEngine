#include "../../include/core/SceneManager.h"

using namespace PhysicsEngine;

SceneManager::SceneManager()
{
	manager = new Manager();
}

SceneManager::~SceneManager()
{
	std::cout << "scene manager calling delete on manager" << std::endl;
	delete manager;
}

void SceneManager::add(Scene scene)
{
	scenes.push_back(scene);

	context.add(scene);
}

void SceneManager::add(AssetFile assetFile)
{
	assetFiles.push_back(assetFile);
}


void SceneManager::init()
{
	loadingSceneIndex = -1;
	activeSceneIndex = -1;
}

bool SceneManager::update()
{
	bool error = false;

	int sceneToLoadIndex = context.getSceneToLoadIndex();
	if(sceneToLoadIndex != activeSceneIndex){
		loadingSceneIndex = sceneToLoadIndex;
		loadingScene = &scenes[sceneToLoadIndex];
	}

	if(loadingScene != NULL){
		error = !manager->load(*loadingScene, assetFiles); 
		if(!error){
			for(int i = 0; i < manager->getNumberOfSystems(); i++){
				System* system = manager->getSystemByIndex(i);

				system->setSceneContext(&context);
				system->init();
			}

			activeSceneIndex = loadingSceneIndex;
			activeScene = loadingScene;
			loadingScene = NULL;
		}
	}

	if(activeScene != NULL && !error){
		for(int i = 0; i < manager->getNumberOfSystems(); i++){
			System* system = manager->getSystemByIndex(i);

			system->update();
		}
	}

	return !error;
}