#include "../../include/core/Input.h"

using namespace PhysicsEngine;

Input global_input = {};

Input& PhysicsEngine::getInput()
{
    return global_input;
}

bool PhysicsEngine::getKey(const Input &input, KeyCode key)
{
    return input.mKeyIsDown[(int)key];
}

bool PhysicsEngine::getKeyDown(const Input &input, KeyCode key)
{
    return (input.mKeyIsDown[(int)key] && !input.mKeyWasDown[(int)key]);
}

bool PhysicsEngine::getKeyUp(const Input &input, KeyCode key)
{
    return (!input.mKeyIsDown[(int)key] && input.mKeyWasDown[(int)key]);
}

bool PhysicsEngine::getMouseButton(const Input &input, MouseButton button)
{
    return input.mMouseButtonIsDown[(int)button];
}

bool PhysicsEngine::getMouseButtonDown(const Input &input, MouseButton button)
{
    return input.mMouseButtonIsDown[(int)button] && !input.mMouseButtonWasDown[(int)button];
}

bool PhysicsEngine::getMouseButtonUp(const Input &input, MouseButton button)
{
    return !input.mMouseButtonIsDown[(int)button] && input.mMouseButtonWasDown[(int)button];
}

bool PhysicsEngine::getXboxButton(const Input &input, XboxButton button)
{
    return input.mXboxButtonIsDown[(int)button];
}

bool PhysicsEngine::getXboxButtonDown(const Input &input, XboxButton button)
{
    return input.mXboxButtonIsDown[(int)button] && !input.mXboxButtonWasDown[(int)button];
}

bool PhysicsEngine::getXboxButtonUp(const Input &input, XboxButton button)
{
    return !input.mXboxButtonIsDown[(int)button] && input.mXboxButtonWasDown[(int)button];
}