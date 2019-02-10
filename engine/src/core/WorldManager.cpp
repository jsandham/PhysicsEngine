#include "../../include/core/WorldManager.h"

using namespace PhysicsEngine;

WorldManager::WorldManager(Scene scene, AssetBundle bundle)
{
	this->scene = scene;
	this->bundle = bundle;

	std::cout << "Scene: " << scene.filepath << " asset bundle: " << bundle.filepath << std::endl;

	world = new World();
}

WorldManager::~WorldManager()
{
	delete world;
}


void WorldManager::init()
{
	if(!world->load(scene, bundle))
	{
		std::cout << "Error: World load failed!" << std::endl;
		return;
	}

	// for(int i = 0; i < world->getNumberOfSystems(); i++){
	// 	System* system = world->getSystemByIndex(i);

	// 	system->init();
	// }
}

bool WorldManager::update(Input input)
{
	// copy Input to world??
	// world->input = input; ???


	// for(int i = 0; i < world->getNumberOfSystems(); i++){
	// 	System* system = world->getSystemByIndex(i);

	// 	system->update(input);
	// 	//system->update();
	// }


	return false;
}




// WorldManager::WorldManager()
// {
// 	world = new World();
// }

// WorldManager::~WorldManager()
// {
// 	delete world;
// }

// void WorldManager::add(Scene scene)
// {
// 	scenes.push_back(scene);

// 	context.add(scene);
// }

// void WorldManager::add(AssetFile assetFile)
// {
// 	assetFiles.push_back(assetFile);
// }


// void WorldManager::init()
// {
// 	loadingSceneIndex = -1;
// 	activeSceneIndex = -1;
// }

// bool WorldManager::update(Input input)
// {
// 	bool error = false;

// 	int sceneToLoadIndex = context.getSceneToLoadIndex();
// 	if(sceneToLoadIndex != activeSceneIndex){
// 		loadingSceneIndex = sceneToLoadIndex;
// 		loadingScene = &scenes[sceneToLoadIndex];
// 	}

// 	if(loadingScene != NULL){
// 		error = !world->load(*loadingScene, assetFiles); 
// 		if(!error){
// 			for(int i = 0; i < world->getNumberOfSystems(); i++){
// 				System* system = world->getSystemByIndex(i);

// 				system->setSceneContext(&context);
// 				system->init();
// 			}

// 			activeSceneIndex = loadingSceneIndex;
// 			activeScene = loadingScene;
// 			loadingScene = NULL;
// 		}
// 	}

// 	if(activeScene != NULL && !error){
// 		for(int i = 0; i < world->getNumberOfSystems(); i++){
// 			System* system = world->getSystemByIndex(i);

// 			system->update(input);
// 		}
// 	}

// 	return !error;
// }