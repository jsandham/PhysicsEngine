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
	mTransform = mCamera->getComponent<Transform>(world);

	mTransform->mPosition = glm::vec3(0, 2, -10);

	//Log::warn(mCamera->getEntityId().toString().c_str());
	//Log::warn(mCamera->getId().toString().c_str());
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
	glm::vec3 position = mTransform->mPosition;
	glm::vec3 front = mTransform->getForward();// mCamera->mFront;
	glm::vec3 up = mTransform->getUp();// mCamera->mUp;
	glm::vec3 right = mTransform->getRight(); //mCamera->mRight;

	//Log::info(("position: " + std::to_string(position.x) + " " + std::to_string(position.y) + " " + std::to_string(position.z) + "\n").c_str());

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
	mTransform->mRotation *= glm::angleAxis(-mouseDelta.x, glm::vec3(0, 1, 0));
	//front = glm::mat3(glm::rotate(glm::mat4(), -mouseDelta.x, up)) * front;

	// rotation around the camera right vector
	mTransform->mRotation *= glm::angleAxis(-mouseDelta.y, glm::vec3(1, 0, 0));
	//front = glm::mat3(glm::rotate(glm::mat4(), -mouseDelta.y, right)) * front;

	//right = glm::normalize(glm::cross(front, up));

	//up = glm::normalize(glm::cross(right, front));


	mCamera->mFrustum.computePlanes(position, front, up, right);
	mCamera->computeViewMatrix(position, front, up);
	//mCamera->updateInternalCameraState();

	mTransform->mPosition = position;
	//mCamera->mFront = front;
	//mCamera->mUp = up;


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
	mTransform->mPosition = glm::vec3(0, 2, -10);
	//mCamera->mFront = glm::vec3(1.0f, 0.0f, 0.0f);
	//mCamera->mUp = glm::vec3(0.0f, 0.0f, 1.0f);
	mCamera->mBackgroundColor = glm::vec4(0.15, 0.15f, 0.15f, 1.0f);
	//mCamera->mFrustum.computePlanes(mTransform->mPosition, mCamera->mFront, mCamera->mUp);
	//mCamera->updateInternalCameraState();
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
