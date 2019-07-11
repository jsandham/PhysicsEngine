#include <iostream>
#include <vector>

#include <core/PoolAllocator.h>
#include <core/World.h>
#include <core/input.h>

#include "../include/LogicSystem.h"

using namespace PhysicsEngine;

LogicSystem::LogicSystem()
{
	
}

LogicSystem::LogicSystem(std::vector<char> data)
{
	deserialize(data);
}

LogicSystem::~LogicSystem()
{

}

std::vector<char> LogicSystem::serialize()
{
	size_t numberOfBytes = sizeof(int);
	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &order, sizeof(int));

	return data;
}

void LogicSystem::deserialize(std::vector<char> data)
{
	order = *reinterpret_cast<int*>(&data[0]);
}

void LogicSystem::init(World* world)
{
	this->world = world;
}

void LogicSystem::update(Input input)
{

}