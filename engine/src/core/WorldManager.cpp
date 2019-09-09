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
	for(size_t i = 0; i < assetFilePaths.size(); i++){
		if(!world.loadAsset(assetFilePaths[i])){
			Log::error("Could not load asset file\n");
			return false;
		}
	}

	if(!world.loadScene(sceneFilePath)){
		Log::error("Could not load scene file\n");
		return false;
	}

	return true;
}

void WorldManager::init()
{
	for(int i = 0; i < world.getNumberOfSystems(); i++){
		System* system = world.getSystemByIndex(i);

		system->init(&world);
	}
}

void WorldManager::update(Time time, Input input)
{
	if(getKeyDown(input, KeyCode::D)){
		world.debug = !world.debug;
	}

	if(world.debug){
		if(getKeyDown(input, KeyCode::NumPad0)){
			world.debugView = 0;
		}
		else if(getKeyDown(input, KeyCode::NumPad1)){
			world.debugView = 1;
		}
		else if(getKeyDown(input, KeyCode::NumPad2)){
			world.debugView = 2;
		}
		else if(getKeyDown(input, KeyCode::NumPad3)){
			world.debugView = 3;
		}
	}

	for(int i = 0; i < world.getNumberOfSystems(); i++){
		System* system = world.getSystemByIndex(i);

		system->update(input);
	}
}