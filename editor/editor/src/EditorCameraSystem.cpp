#include "../include/EditorCameraSystem.h"
#include "../include/EditorOnlyEntityCreation.h"

#include <components/Camera.h>

#include <core/PoolAllocator.h>
#include <core/Input.h>
#include <core/Time.h>
#include <core/Log.h>
#include <core/World.h>

using namespace PhysicsEngine;

const float EditorCameraSystem::YAW_PAN_SENSITIVITY = 0.0025f;
const float EditorCameraSystem::PITCH_PAN_SENSITIVITY = 0.0025f;
const float EditorCameraSystem::ZOOM_SENSITIVITY = 125.0f;
const float EditorCameraSystem::TRANSLATE_SENSITIVITY = 75.0f;

EditorCameraSystem::EditorCameraSystem()
{
	mTransform = NULL;
	mCamera = NULL;

	mMousePosX = 0;
	mMousePosY = 0;
	mIsLeftMouseClicked = false;
	mIsRightMouseClicked = false;
	rotationOnClick = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
}

EditorCameraSystem::EditorCameraSystem(std::vector<char> data)
{
	deserialize(data);

	mMousePosX = 0;
	mMousePosY = 0;
	mIsLeftMouseClicked = false;
	mIsRightMouseClicked = false;
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
	EditorCameraSystemHeader header;
	header.mSystemId = systemId;
	header.mUpdateOrder = static_cast<int32_t>(mOrder);

	std::vector<char> data(sizeof(EditorCameraSystemHeader));

	memcpy(&data[0], &header, sizeof(EditorCameraSystemHeader));

	return data;
}

void EditorCameraSystem::deserialize(const std::vector<char>& data)
{
	const EditorCameraSystemHeader* header = reinterpret_cast<const EditorCameraSystemHeader*>(&data[0]);

	mSystemId = header->mSystemId;
	mOrder = static_cast<int>(header->mUpdateOrder);
}

void EditorCameraSystem::init(World* world)
{
	mWorld = world;

	mCamera = world->getComponentByIndex<Camera>(0);
	mTransform = mCamera->getComponent<Transform>(world);

	mTransform->mPosition = glm::vec3(0, 2, -10);
}

void EditorCameraSystem::update(Input input, Time time)
{
	glm::vec3 position = mTransform->mPosition;
	glm::vec3 front = mTransform->getForward();
	glm::vec3 up = mTransform->getUp();
	glm::vec3 right = mTransform->getRight();

	// D pad controls
	if (!getMouseButton(input, RButton)) {
		if (getKey(input, KeyCode::Up)) {
			position += EditorCameraSystem::ZOOM_SENSITIVITY * time.deltaTime * front;
		}
		if (getKey(input, KeyCode::Down)) {
			position -= EditorCameraSystem::ZOOM_SENSITIVITY * time.deltaTime * front;
		}
		if (getKey(input, KeyCode::Left)) {
			position += EditorCameraSystem::TRANSLATE_SENSITIVITY * time.deltaTime * right;
		}
		if (getKey(input, KeyCode::Right)) {
			position -= EditorCameraSystem::TRANSLATE_SENSITIVITY * time.deltaTime * right;
		}
	}

	// WASD controls
	if (getMouseButton(input, RButton)) {
		if (getKey(input, KeyCode::W)) {
			position += EditorCameraSystem::ZOOM_SENSITIVITY * time.deltaTime * front;
		}
		if (getKey(input, KeyCode::S)) {
			position -= EditorCameraSystem::ZOOM_SENSITIVITY * time.deltaTime * front;
		}
		if (getKey(input, KeyCode::A)) {
			position += EditorCameraSystem::TRANSLATE_SENSITIVITY * time.deltaTime * right;
		}
		if (getKey(input, KeyCode::D)) {
			position -= EditorCameraSystem::TRANSLATE_SENSITIVITY * time.deltaTime * right;
		}
	}

	// Mouse scroll wheel
	position += EditorCameraSystem::ZOOM_SENSITIVITY * input.mouseDelta * time.deltaTime * front;

	// Mouse position
	mMousePosX = input.mousePosX;
	mMousePosY = input.mousePosY;

	// Mouse buttons
	if (getMouseButtonDown(input, RButton)) {
		mMousePosXOnRightClick = mMousePosX;
		mMousePosYOnRightClick = mMousePosY;
		rotationOnClick = mTransform->mRotation;
	}
	else if (getMouseButton(input, RButton)) {
		float yaw = EditorCameraSystem::YAW_PAN_SENSITIVITY * (mMousePosXOnRightClick - mMousePosX);
		float pitch = EditorCameraSystem::PITCH_PAN_SENSITIVITY * (mMousePosYOnRightClick - mMousePosY);

		// https://gamedev.stackexchange.com/questions/136174/im-rotating-an-object-on-two-axes-so-why-does-it-keep-twisting-around-the-thir
		mTransform->mRotation = glm::angleAxis(yaw, glm::vec3(0, 1, 0)) * rotationOnClick * glm::angleAxis(pitch, glm::vec3(1, 0, 0));
	}

	mIsLeftMouseClicked = false;
	mIsRightMouseClicked = false;
	if (getMouseButtonDown(input, LButton)){
		mIsLeftMouseClicked = true;
	}
	if (getMouseButtonDown(input, RButton)) {
		mIsRightMouseClicked = true;
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

void EditorCameraSystem::setRenderPath(RenderPath path)
{
	mCamera->mRenderPath = path;
}

void EditorCameraSystem::setSSAO(CameraSSAO ssao)
{
	mCamera->mSSAO = ssao;
}

Viewport EditorCameraSystem::getViewport() const
{
	return mCamera->mViewport;
}

Frustum EditorCameraSystem::getFrustum() const
{
	return mCamera->mFrustum;
}

RenderPath EditorCameraSystem::getRenderPath() const
{
	return mCamera->mRenderPath;
}

CameraSSAO EditorCameraSystem::getSSAO() const
{
	return mCamera->mSSAO;
}

int EditorCameraSystem::getMousePosX() const
{
	return mMousePosX;
}

int EditorCameraSystem::getMousePosY() const
{
	return mMousePosY;
}

bool EditorCameraSystem::isLeftMouseClicked() const
{
	return mIsLeftMouseClicked;
}

bool EditorCameraSystem::isRightMouseClicked() const
{
	return mIsRightMouseClicked;
}

Guid EditorCameraSystem::getMeshRendererUnderMouse(float nx, float ny) const
{
	int x = mCamera->mViewport.mX + mCamera->mViewport.mWidth * nx;
	int y = mCamera->mViewport.mY + mCamera->mViewport.mHeight * ny;

	return mCamera->getMeshRendererIdAtScreenPos(x, y);
	/*return mCamera->getMeshRendererIdAtScreenPos(mMousePosX, mMousePosY);*/
}

GLuint EditorCameraSystem::getNativeGraphicsColorTex() const
{
	return mCamera->getNativeGraphicsColorTex();
}

GLuint EditorCameraSystem::getNativeGraphicsDepthTex() const
{
	return mCamera->getNativeGraphicsDepthTex();
}

GLuint EditorCameraSystem::getNativeGraphicsColorPickingTex() const
{
	return mCamera->getNativeGraphicsColorPickingTex();
}

GLuint EditorCameraSystem::getNativeGraphicsPositionTex() const
{
	return mCamera->getNativeGraphicsPositionTex();
}

GLuint EditorCameraSystem::getNativeGraphicsNormalTex() const
{
	return mCamera->getNativeGraphicsNormalTex();
}

GLuint EditorCameraSystem::getNativeGraphicsAlbedoSpecTex() const
{
	return mCamera->getNativeGraphicsAlbedoSpecTex();
}

GLuint EditorCameraSystem::getNativeGraphicsSSAOColorTex() const
{
	return mCamera->getNativeGraphicsSSAOColorTex();
}

GLuint EditorCameraSystem::getNativeGraphicsSSAONoiseTex() const
{
	return mCamera->getNativeGraphicsSSAONoiseTex();
}

GraphicsQuery EditorCameraSystem::getQuery() const
{
	return mCamera->mQuery;
}
