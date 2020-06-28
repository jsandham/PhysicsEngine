#include <iostream>
#include <vector>
#include <unordered_map>

#include <core/Load.h>
#include <core/PoolAllocator.h>

using namespace PhysicsEngine;

Asset* PhysicsEngine::loadAsset(std::unordered_map<int, Allocator*>* allocatorMap, std::vector<char> data, int type, int* index)
{
	std::cout << "Error: Invalid asset type (" << type << ") when trying to load asset" << std::endl;
	return NULL;
}

Component* PhysicsEngine::loadComponent(std::unordered_map<int, Allocator*>* allocatorMap, std::vector<char> data, int type, int* index)
{
	std::cout << "Error: Invalid component type (" << type << ") when trying to load component" << std::endl;
	return NULL;
}

System* PhysicsEngine::loadSystem(std::unordered_map<int, Allocator*>* allocatorMap, std::vector<char> data, int type, int* index)
{
	return NULL;
}

Component* PhysicsEngine::destroyComponent(std::unordered_map<int, Allocator*>* allocatorMap, int type, int index)
{
	return NULL;
}