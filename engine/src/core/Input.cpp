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
	return input.buttonIsDown[(int)button];
}

bool PhysicsEngine::getMouseButtonDown(Input input, MouseButton button)
{
	return input.buttonIsDown[(int)button] && !input.buttonWasDown[(int)button];
}

bool PhysicsEngine::getMouseButtonUp(Input input, MouseButton button)
{
	return !input.buttonIsDown[(int)button] && input.buttonWasDown[(int)button];
}