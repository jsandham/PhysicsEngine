#include <iostream>
#include <vector>

#include <core/Load.h>
#include <core/PoolAllocator.h>

using namespace PhysicsEngine;

Asset* PhysicsEngine::loadAsset(std::vector<char> data, int type, int* index)
{
	std::cout << "Error: Invalid asset type (" << type << ") when trying to load asset" << std::endl;
	return NULL;
}

Component* PhysicsEngine::loadComponent(std::vector<char> data, int type, int* index)
{
	std::cout << "Error: Invalid component type (" << type << ") when trying to load component" << std::endl;
	return NULL;
}

System* PhysicsEngine::loadSystem(std::vector<char> data, int type, int* index)
{
	return NULL;
}

Component* PhysicsEngine::destroyComponent(int type, int index)
{
	return NULL;
}