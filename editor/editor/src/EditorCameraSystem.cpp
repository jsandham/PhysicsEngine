#include "../include/EditorCameraSystem.h"

#include <components/Camera.h>

#include <core/PoolAllocator.h>
#include <core/Input.h>
#include <core/Time.h>
#include <core/Log.h>
#include <core/World.h>

using namespace PhysicsEngine;

const float EditorCameraSystem::YAW_PAN_SENSITIVITY = 0.0025f;
const float EditorCameraSystem::PITCH_PAN_SENSITIVITY = 0.0025f;
const float EditorCameraSystem::ZOOM_SENSITIVITY = 0.5f;
const float EditorCameraSystem::TRANSLATE_SENSITIVITY = 0.075f;

EditorCameraSystem::EditorCameraSystem()
{
	mTransform = NULL;
	mCamera = NULL;

	mCurrentPosX = 0;
	mCurrentPosY = 0;
	rotationOnClick = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
}

EditorCameraSystem::EditorCameraSystem(std::vector<char> data)
{
	deserialize(data);

	mCurrentPosX = 0;
	mCurrentPosY = 0;
}

EditorCameraSystem::~EditorCameraSystem()
{

}

std::vector<char> EditorCameraSystem::serialize() const
{
	return serialize(mSystemId);
}

std::vector<char> EditorCameraSystem::serialize(Guid systemId) const
{
	std::vector<char> data(sizeof(int));

	memcpy(&data[0], &mOrder, sizeof(int));

	return data;
}

void EditorCameraSystem::deserialize(std::vector<char> data)
{
	mOrder = *reinterpret_cast<int*>(&data[0]);
}

void EditorCameraSystem::init(World* world)
{
	mWorld = world;

	mCamera = world->createEditorCamera();
	mTransform = mCamera->getComponent<Transform>(world);

	mTransform->mPosition = glm::vec3(0, 2, -10);
}

void EditorCameraSystem::update(Input input)
{
	glm::vec3 position = mTransform->mPosition;
	glm::vec3 front = mTransform->getForward();
	glm::vec3 up = mTransform->getUp();
	glm::vec3 right = mTransform->getRight();

	//Log::info(("position: " + std::to_string(position.x) + " " + std::to_string(position.y) + " " + std::to_string(position.z) + "\n").c_str());

	// D pad controls
	if (!getMouseButton(input, RButton)) {
		if (getKey(input, KeyCode::Up)) {
			position += EditorCameraSystem::ZOOM_SENSITIVITY * front;
		}
		if (getKey(input, KeyCode::Down)) {
			position -= EditorCameraSystem::ZOOM_SENSITIVITY * front;
		}
		if (getKey(input, KeyCode::Left)) {
			position -= right * EditorCameraSystem::TRANSLATE_SENSITIVITY;
		}
		if (getKey(input, KeyCode::Right)) {
			position += right * EditorCameraSystem::TRANSLATE_SENSITIVITY;
		}
	}

	// WASD controls
	if (getMouseButton(input, RButton)) {
		if (getKey(input, KeyCode::W)) {
			position += EditorCameraSystem::ZOOM_SENSITIVITY * front;
		}
		if (getKey(input, KeyCode::S)) {
			position -= EditorCameraSystem::ZOOM_SENSITIVITY * front;
		}
		if (getKey(input, KeyCode::A)) {
			position -= right * EditorCameraSystem::TRANSLATE_SENSITIVITY;
		}
		if (getKey(input, KeyCode::D)) {
			position += right * EditorCameraSystem::TRANSLATE_SENSITIVITY;
		}
	}

	// Mouse scroll wheel
	glm::vec2 mouseDelta = glm::vec2(0.0f, 0.0f);

	position += EditorCameraSystem::ZOOM_SENSITIVITY * input.mouseDelta * front;

	// Mouse buttons
	if (getMouseButtonDown(input, RButton)) {
		mCurrentPosX = input.mousePosX;
		mCurrentPosY = input.mousePosY;
		rotationOnClick = mTransform->mRotation;
	}
	else if (getMouseButton(input, RButton)) {
		mouseDelta.x = EditorCameraSystem::YAW_PAN_SENSITIVITY * (input.mousePosX - mCurrentPosX);
		mouseDelta.y = EditorCameraSystem::PITCH_PAN_SENSITIVITY * (input.mousePosY - mCurrentPosY);

		//https://gamedev.stackexchange.com/questions/136174/im-rotating-an-object-on-two-axes-so-why-does-it-keep-twisting-around-the-thir
		mTransform->mRotation = glm::angleAxis(mouseDelta.x, glm::vec3(0, 1, 0)) * rotationOnClick * glm::angleAxis(mouseDelta.y, glm::vec3(1, 0, 0));
	}

	mCamera->mFrustum.computePlanes(position, front, up, right);
	mCamera->computeViewMatrix(position, front, up);

	mTransform->mPosition = position;
}

void EditorCameraSystem::resetCamera()
{
	mTransform->mPosition = glm::vec3(0, 2, -10);
	mCamera->mBackgroundColor = glm::vec4(0.15, 0.15f, 0.15f, 1.0f);
}

void EditorCameraSystem::setViewport(Viewport viewport)
{
	mCamera->mViewport = viewport;
}

void EditorCameraSystem::setFrustum(Frustum frustum)
{
	mCamera->mFrustum = frustum;
}

Viewport EditorCameraSystem::getViewport() const
{
	return mCamera->mViewport;
}

Frustum EditorCameraSystem::getFrustum() const
{
	return mCamera->mFrustum;
}
