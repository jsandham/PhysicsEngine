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

	for(int i = 0; i < world->getNumberOfSystems(); i++){
		System* system = world->getSystemByIndex(i);

		system->init(world);

		std::cout << "initializing system type: " << system->getType() << " with order: " << system->getOrder() << std::endl;
	}
}

bool WorldManager::update(Input input)
{
	for(int i = 0; i < world->getNumberOfSystems(); i++){
		System* system = world->getSystemByIndex(i);

		system->update(input);
	}


	return true;
}