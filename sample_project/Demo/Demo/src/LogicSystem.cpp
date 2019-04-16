#include <iostream>
#include <vector>

#include <core/PoolAllocator.h>
#include <core/World.h>
#include <core/input.h>

#include "../include/LogicSystem.h"

using namespace PhysicsEngine;

LogicSystem::LogicSystem()
{
	type = 20;
}

LogicSystem::LogicSystem(std::vector<char> data)
{
	size_t index = sizeof(char);
	type = *reinterpret_cast<int*>(&data[index]);
	index += sizeof(int);
	order = *reinterpret_cast<int*>(&data[index]);

	if (type != 20){
		std::cout << "Error: System type (" << type << ") found in data array is invalid" << std::endl;
	}
}

LogicSystem::~LogicSystem()
{

}

//void* LogicSystem::operator new(size_t size)
//{
//	return getAllocator<LogicSystem>().allocate();
//}
//
//void LogicSystem::operator delete(void*)
//{
//
//}

void LogicSystem::init(World* world)
{
	this->world = world;
}

void LogicSystem::update(Input input)
{

}