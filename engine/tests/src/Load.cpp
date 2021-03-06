#include <iostream>
#include <vector>
#include <unordered_map>

#include <core/Load.h>
#include <core/PoolAllocator.h>
#include <systems/System.h>
#include <components/Component.h>

using namespace PhysicsEngine;

void PhysicsEngine::addAssetIdToIndexMap(std::unordered_map<Guid, int>* idToGlobalIndex, std::unordered_map<Guid, int>* idToType, const Guid& id, int type, int index)
{

}

void PhysicsEngine::addComponentIdToIndexMap(std::unordered_map<Guid, int>* idToGlobalIndex, std::unordered_map<Guid, int>* idToType, const Guid& id, int type, int index)
{

}

void PhysicsEngine::addSystemIdToIndexMap(std::unordered_map<Guid, int>* idToGlobalIndex, std::unordered_map<Guid, int>* idToType, const Guid& id, int type, int index)
{

}

void PhysicsEngine::removeAssetIdFromIndexMap(std::unordered_map<Guid, int>* idToGlobalIndex, std::unordered_map<Guid, int>* idToType, const Guid& id, int type)
{

}

void PhysicsEngine::removeComponentIdFromIndexMap(std::unordered_map<Guid, int>* idToGlobalIndex, std::unordered_map<Guid, int>* idToType, const Guid& id, int type)
{

}

void PhysicsEngine::removeSystemIdFromIndexMap(std::unordered_map<Guid, int>* idToGlobalIndex, std::unordered_map<Guid, int>* idToType, const Guid& id, int type)
{
	
}

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