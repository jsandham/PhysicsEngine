#include "../../include/core/WorldManager.h"

using namespace PhysicsEngine;

WorldManager::WorldManager(Scene scene, AssetBundle bundle)
{
	this->scene = scene;
	this->bundle = bundle;

	std::cout << "Scene: " << scene.filepath << " asset bundle: " << bundle.filepath << std::endl;
}

WorldManager::~WorldManager()
{
}


void WorldManager::init()
{
	if(!world.load(scene, bundle))
	{
		std::cout << "Error: World load failed!" << std::endl;
		return;
	}

	for(int i = 0; i < world.getNumberOfSystems(); i++){
		System* system = world.getSystemByIndex(i);

		system->init(&world);

		std::cout << "initializing system type: " << system->getType() << " with order: " << system->getOrder() << std::endl;
	}
}

bool WorldManager::update(Time time, Input input)
{
	//std::cout << "Time: " << time.deltaTime << " " << getFPS(time) << std::endl;

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


	return true;
}