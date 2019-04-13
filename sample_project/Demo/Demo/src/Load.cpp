#include <iostream>
#include <vector>

#include <core/Load.h>
#include <core/PoolAllocator.h>

#include "../include/LogicSystem.h"
#include "../include/PlayerSystem.h"

using namespace PhysicsEngine;

Asset* PhysicsEngine::loadAsset(std::vector<char> data, int* index)
{
	int type = *reinterpret_cast<int*>(&data[0]);

	*index = -1;

	std::cout << "Error: Invalid asset type (" << type << ") when trying to load asset" << std::endl;
	return NULL;
}

Component* PhysicsEngine::loadComponent(std::vector<char> data, int* index, int* instanceType)
{
	int type = *reinterpret_cast<int*>(&data[sizeof(char)]);

	*index = -1;
	*instanceType = -1;// Component::getInstanceType<>();

	std::cout << "Error: Invalid component type (" << type << ") when trying to load component" << std::endl;
	return NULL;
}

System* PhysicsEngine::loadSystem(std::vector<char> data, int* index)
{
	int type = *reinterpret_cast<int*>(&data[sizeof(char)]);

	//std::cout << "load internal system called system of type: " << type << std::endl;

	if (type == 20){
		*index = (int)getAllocator<LogicSystem>().getCount();
		return new LogicSystem(data);
	}
	else if (type == 21){
		*index = (int)getAllocator<PlayerSystem>().getCount();
		return new PlayerSystem(data);
	}
	else{
		std::cout << "Error: Invalid system type (" << type << ") when trying to load system" << std::endl;
		return NULL;
	}
}