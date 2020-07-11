#include <iostream>
#include <vector>
#include <unordered_map>

#include <core/Load.h>
#include <core/PoolAllocator.h>

using namespace PhysicsEngine;

Asset* PhysicsEngine::getAsset(std::unordered_map<int, Allocator*>* allocatorMap, int type, int index)
{
	return NULL;
}

Component* PhysicsEngine::getComponent(std::unordered_map<int, Allocator*>* allocatorMap, int type, int index)
{
	return NULL;
}

System* PhysicsEngine::getSystem(std::unordered_map<int, Allocator*>* allocatorMap, int type, int index)
{
	return NULL;
}

Asset* PhysicsEngine::loadAsset(std::unordered_map<int, Allocator*>* allocatorMap, const std::vector<char>& data, int type, int* index)
{
	return NULL;
}

Component* PhysicsEngine::loadComponent(std::unordered_map<int, Allocator*>* allocatorMap, const std::vector<char>& data, int type, int* index)
{
	return NULL;
}

System* PhysicsEngine::loadSystem(std::unordered_map<int, Allocator*>* allocatorMap, const std::vector<char>& data, int type, int* index)
{
	return NULL;
}

Component* PhysicsEngine::destroyComponent(std::unordered_map<int, Allocator*>* allocatorMap, int type, int index)
{
	return NULL;
}