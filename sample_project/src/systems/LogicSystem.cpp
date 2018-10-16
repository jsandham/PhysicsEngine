#include <iostream>

#include "../include/systems/LogicSystem.h"

using namespace PhysicsEngine;

LogicSystem::LogicSystem()
{
	type = 10;
}

LogicSystem::LogicSystem(unsigned char* data)
{
	type = 10;
}

LogicSystem::~LogicSystem()
{

}

void LogicSystem::init()
{
	std::cout << "LogicSystem init called" << std::endl;
}

void LogicSystem::update()
{
	//std::cout << "LogicSystem update called" << std::endl;
}