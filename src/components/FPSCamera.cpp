#include "FPSCamera.h"
#include "../Input.h"
#include "../Time.h"

using namespace PhysicsEngine;

const GLfloat FPSCamera::PAN_SENSITIVITY = 0.01f;
const GLfloat FPSCamera::SCROLL_SENSITIVITY = 0.1f;
const GLfloat FPSCamera::TRANSLATE_SENSITIVITY = 0.05f;

FPSCamera::FPSCamera() : Camera()
{
	firstFrame = true;

	yaw = 0.0;
	pitch = 0.0;

	jumpInProgress = false;

	mouseDelta = glm::vec2(0.0f);
}

FPSCamera::~FPSCamera()
{

}

bool FPSCamera::getFirstFrame()
{
	return firstFrame;
}

bool FPSCamera::getJumpInProgress()
{
	return jumpInProgress;
}

float FPSCamera::getYaw()
{
	return yaw;
}

float FPSCamera::getPitch()
{
	return pitch;
}

float FPSCamera::getJumpTime()
{
	return jumpTime;
}

void FPSCamera::setFirstFrame(bool firstFrame)
{
	this->firstFrame = firstFrame;
}

void FPSCamera::setJumpInProgress(bool jumpInProgress)
{
	this->jumpInProgress = jumpInProgress;
}

void FPSCamera::setYaw(float yaw)
{
	this->yaw = yaw;
}

void FPSCamera::setPitch(float pitch)
{
	this->pitch = pitch;
}

void FPSCamera::setJumpTime(float jumpTime)
{
	this->jumpTime = jumpTime;
}


glm::vec2& FPSCamera::getMouseDelta()
{
	return mouseDelta;
}

void FPSCamera::setMouseDelta(glm::vec2& mouseDelta)
{
	this->mouseDelta = mouseDelta;
}
