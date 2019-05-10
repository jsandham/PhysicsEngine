#include "../../include/core/Input.h"

using namespace PhysicsEngine;

bool PhysicsEngine::getKey(Input input, KeyCode key)
{
	return input.keyIsDown[(int)key];
}

bool PhysicsEngine::getKeyDown(Input input, KeyCode key)
{
	return (input.keyIsDown[(int)key] && !input.keyWasDown[(int)key]);
}

bool PhysicsEngine::getKeyUp(Input input, KeyCode key)
{
	return (!input.keyIsDown[(int)key] && input.keyWasDown[(int)key]);
}

bool PhysicsEngine::getMouseButton(Input input, MouseButton button)
{
	return input.mouseButtonIsDown[(int)button];
}

bool PhysicsEngine::getMouseButtonDown(Input input, MouseButton button)
{
	return input.mouseButtonIsDown[(int)button] && !input.mouseButtonWasDown[(int)button];
}

bool PhysicsEngine::getMouseButtonUp(Input input, MouseButton button)
{
	return !input.mouseButtonIsDown[(int)button] && input.mouseButtonWasDown[(int)button];
}

bool PhysicsEngine::getXboxButton(Input input, XboxButton button)
{
	return input.xboxButtonIsDown[(int)button];
}

bool PhysicsEngine::getXboxButtonDown(Input input, XboxButton button)
{
	return input.xboxButtonIsDown[(int)button] && !input.xboxButtonWasDown[(int)button];
}

bool PhysicsEngine::getXboxButtonUp(Input input, XboxButton button)
{
	return !input.xboxButtonIsDown[(int)button] && input.xboxButtonWasDown[(int)button];
}