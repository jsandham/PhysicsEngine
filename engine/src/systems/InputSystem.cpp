#include <iostream>
#include "../../include/systems/InputSystem.h"

using namespace PhysicsEngine;

InputSystem::InputSystem()
{

}

InputSystem::~InputSystem()
{

}

void InputSystem::init(World* world)
{
	this->world = world;
}

void InputSystem::update(Input input)
{

}