#include "../../include/systems/CleanUpSystem.h"

#include "../../include/core/World.h"

using namespace PhysicsEngine;

CleanUpSystem::CleanUpSystem(unsigned char* data)
{
	type = 2;
}

CleanUpSystem::~CleanUpSystem()
{

}

void CleanUpSystem::init()
{

}

void CleanUpSystem::update(Input input)
{
	// std::vector<int> entitiesMarkedForLatentDestroy;
	// for(int i = 0; i < entitiesMarkedForLatentDestroy.size(); i++){
	// 	manager->immediateDestroy(entitiesMarkedForLatentDestroy[i]):
	// }
}