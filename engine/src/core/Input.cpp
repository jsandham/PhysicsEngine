#include <iostream>

#include "../../include/core/Input.h"

using namespace PhysicsEngine;

std::vector<bool> Input::keyIsDown(41, false);
std::vector<bool> Input::keyWasDown(41, false);
std::vector<bool> Input::buttonIsDown(3, false);
std::vector<bool> Input::buttonWasDown(3, false);
int Input::mousePosX = 0;
int Input::mousePosY = 0;
int Input::mouseDelta = 0;

bool Input::getKey(KeyCode key)
{
	return keyIsDown[(int)key];
}

bool Input::getKeyDown(KeyCode key)
{
	return (keyIsDown[(int)key] && !keyWasDown[(int)key]);
}

bool Input::getKeyUp(KeyCode key)
{
	return (!keyIsDown[(int)key] && keyWasDown[(int)key]);
}

bool Input::getMouseButton(MouseButton button)
{
	return buttonIsDown[(int)button];
}

bool Input::getMouseButtonDown(MouseButton button)
{
	return buttonIsDown[(int)button] && !buttonWasDown[(int)button];
}

bool Input::getMouseButtonUp(MouseButton button)
{
	return !buttonIsDown[(int)button] && buttonWasDown[(int)button];
}

int Input::getMousePosX()
{
	return mousePosX;
}

int Input::getMousePosY()
{
	return mousePosY;
}

int Input::getMouseDelta()
{
	return mouseDelta;
}

void Input::setKeyState(KeyCode key, bool isDown, bool wasDown)
{
	keyIsDown[(int)key] = isDown;
	keyWasDown[(int)key] = wasDown;
}

void Input::setMouseButtonState(MouseButton button, bool isDown, bool wasDown)
{
	buttonIsDown[(int)button] = isDown;
	buttonWasDown[(int)button] = wasDown;
}

void Input::setMousePosition(int x, int y)
{
	mousePosX = x;
	mousePosY = y;
}

void Input::setMouseDelta(int delta)
{
	mouseDelta = delta;
}

void Input::updateEOF()
{
	for(unsigned int i = 0; i < 3; i++){
		buttonWasDown[i] = buttonIsDown[i];
	}

	mouseDelta = 0;

	for(unsigned int i = 0; i < keyIsDown.size(); i++){
		keyIsDown[i] = false;
		keyWasDown[i] = false;
	}
}