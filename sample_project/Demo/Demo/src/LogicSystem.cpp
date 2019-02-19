#include <iostream>
#include <vector>

#include <core/PoolAllocator.h>
#include <core/World.h>
#include <core/input.h>

#include "../include/LogicSystem.h"

using namespace PhysicsEngine;

LogicSystem::LogicSystem()
{
	type = 10;
}

LogicSystem::LogicSystem(std::vector<char> data)
{
	type = 10;
}

LogicSystem::~LogicSystem()
{

}

void* LogicSystem::operator new(size_t size)
{
	return getAllocator<LogicSystem>().allocate();
}

void LogicSystem::operator delete(void*)
{

}

void LogicSystem::init(World* world)
{
	this->world = world;
}

void LogicSystem::update(Input input)
{

}