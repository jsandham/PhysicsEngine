#include "../include/EditorCameraSystem.h"

#include <components/Camera.h>

#include <core/PoolAllocator.h>
#include <core/Input.h>
#include <core/Time.h>
#include <core/Log.h>
#include <core/World.h>

using namespace PhysicsEngine;

const float EditorCameraSystem::PAN_SENSITIVITY = 0.001f;
const float EditorCameraSystem::SCROLL_SENSITIVITY = 0.5f;
const float EditorCameraSystem::TRANSLATE_SENSITIVITY = 0.05f;

EditorCameraSystem::EditorCameraSystem()
{

}

EditorCameraSystem::EditorCameraSystem(std::vector<char> data)
{
	deserialize(data);

	mLastPosX = 0;
	mLastPosY = 0;
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

	Log::warn(mCamera->getEntityId().toString().c_str());
	Log::warn(mCamera->getId().toString().c_str());
}

void EditorCameraSystem::update(Input input)
{
	/*camera = world->getComponentByIndex<Camera>(0);
	if (camera == NULL) {
		return;
	}*/

	/*glm::vec3 position = camera->getPosition();
	glm::vec3 front = camera->getFront();
	glm::vec3 up = camera->getUp();
	glm::vec3 right = camera->getRight();

	int lastPosX = camera->lastPosX;
	int lastPosY = camera->lastPosY;

	int currentPosX = camera->currentPosX;
	int currentPosY = camera->currentPosY;*/
	glm::vec3 position = mCamera->mPosition;
	glm::vec3 front = mCamera->mFront;
	glm::vec3 up = mCamera->mUp;
	glm::vec3 right = mCamera->mRight;

	if (getKey(input, KeyCode::Up)) {
		position += up * EditorCameraSystem::TRANSLATE_SENSITIVITY;
	}
	if (getKey(input, KeyCode::Down)) {
		position -= up * EditorCameraSystem::TRANSLATE_SENSITIVITY;
	}
	if (getKey(input, KeyCode::Left)) {
		position -= right * EditorCameraSystem::TRANSLATE_SENSITIVITY;
	}
	if (getKey(input, KeyCode::Right)) {
		position += right * EditorCameraSystem::TRANSLATE_SENSITIVITY;
	}

	glm::vec2 mouseDelta = glm::vec2(0.0f, 0.0f);

	position += EditorCameraSystem::SCROLL_SENSITIVITY * input.mouseDelta * front;

	if (getMouseButtonDown(input, LButton)) {
		mCurrentPosX = input.mousePosX;
		mCurrentPosY = input.mousePosY;
	}
	else if (getMouseButton(input, LButton)) {
		mCurrentPosX = input.mousePosX;
		mCurrentPosY = input.mousePosY;
		mouseDelta.x = EditorCameraSystem::PAN_SENSITIVITY * (mCurrentPosX - mLastPosX);
		mouseDelta.y = EditorCameraSystem::PAN_SENSITIVITY * (mCurrentPosY - mLastPosY);
	}
	else if (getMouseButtonUp(input, LButton)) {
		mouseDelta = glm::vec2(0.0f, 0.0f);
	}

	mLastPosX = mCurrentPosX;
	mLastPosY = mCurrentPosY;

	// rotation around the camera up vector
	front = glm::mat3(glm::rotate(glm::mat4(), -mouseDelta.x, up)) * front;
	front = glm::normalize(front);

	// rotation around the camera right vector
	front = glm::mat3(glm::rotate(glm::mat4(), -mouseDelta.y, right)) * front;
	front = glm::normalize(front);

	right = glm::normalize(glm::cross(front, up));
	//right = glm::normalize(glm::cross(front, worldUp));

	up = glm::normalize(glm::cross(right, front));

	/*camera->setPosition(position);
	camera->setFront(front);
	camera->setUp(up);
	camera->setRight(right);

	camera->lastPosX = lastPosX;
	camera->lastPosY = lastPosY;
	camera->currentPosX = currentPosX;
	camera->currentPosY = currentPosY;*/
	mCamera->mPosition = position;
	mCamera->mFront = front;
	mCamera->mUp = up;


	//camera->lastPosX = lastPosX;
	//camera->lastPosY = lastPosY;
	//camera->currentPosX = currentPosX;
	//camera->currentPosY = currentPosY;

	mCamera->updateInternalCameraState();




	if (getKeyDown(input, KeyCode::A)) {
		std::cout << "Key code A down" << std::endl;
	}
	if (getKey(input, KeyCode::B)) {
		std::cout << "Key code B" << std::endl;
	}
	if (getKeyUp(input, KeyCode::C)) {
		std::cout << "Key code C" << std::endl;
	}
}

void EditorCameraSystem::resetCamera()
{
	mCamera->mPosition = glm::vec3(0.0f, 0.0f, 1.0f);
	mCamera->mFront = glm::vec3(1.0f, 0.0f, 0.0f);
	mCamera->mUp = glm::vec3(0.0f, 0.0f, 1.0f);
	mCamera->mBackgroundColor = glm::vec4(0.15, 0.15f, 0.15f, 1.0f);
	mCamera->updateInternalCameraState();
}
