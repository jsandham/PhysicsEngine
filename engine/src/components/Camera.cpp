#include "../../include/components/Camera.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Camera::Camera()
{
	mComponentId = Guid::INVALID;
	mEntityId = Guid::INVALID;
	mTargetTextureId = Guid::INVALID;

	mMainFBO = 0;
	mColorTex = 0;
	mDepthTex = 0;

	mGeometryFBO = 0;
	mPositionTex = 0;
	mNormalTex = 0;

	mMode = CameraMode::Main;
	mSSAO = CameraSSAO::SSAO_Off;

	mViewport.mX = 0;
	mViewport.mY = 0;
	mViewport.mWidth = 1024;
	mViewport.mHeight = 1024;

	mBackgroundColor = glm::vec4(0.15f, 0.15f, 0.15f, 1.0f);

	mFrustum.mFov = 45.0f;
	mFrustum.mAspectRatio = 1.0f;
	mFrustum.mNearPlane = 0.1f;
	mFrustum.mFarPlane = 250.0f;

	mIsCreated = false;
}

Camera::Camera(std::vector<char> data)
{
	deserialize(data);
}

Camera::~Camera()
{

}

std::vector<char> Camera::serialize() const
{
	return serialize(mComponentId, mEntityId);
}

std::vector<char> Camera::serialize(Guid componentId, Guid entityId) const
{
	CameraHeader header;
	header.mComponentId = componentId;
	header.mEntityId = entityId;
	header.mTargetTextureId = mTargetTextureId;
	header.mMode = mMode;
	header.mSSAO = mSSAO;
	header.mBackgroundColor = mBackgroundColor;
	header.mX = mViewport.mX;
	header.mY = mViewport.mY;
	header.mWidth = mViewport.mWidth;
	header.mHeight = mViewport.mHeight;
	header.mFov = mFrustum.mFov;
	header.mAspectRatio = mFrustum.mAspectRatio;
	header.mNearPlane = mFrustum.mNearPlane;
	header.mFarPlane = mFrustum.mFarPlane;

	std::vector<char> data(sizeof(CameraHeader));

	memcpy(&data[0], &header, sizeof(CameraHeader));

	return data;
}

void Camera::deserialize(std::vector<char> data)
{
	CameraHeader* header = reinterpret_cast<CameraHeader*>(&data[0]);

	mComponentId = header->mComponentId;
	mEntityId = header->mEntityId;
	mTargetTextureId = header->mTargetTextureId;

	mMode = header->mMode;
	mSSAO = header->mSSAO;

	mViewport.mX = header->mX;
	mViewport.mY = header->mY;
	mViewport.mWidth = header->mWidth;
	mViewport.mHeight = header->mHeight;

	mFrustum.mFov = header->mFov;
	mFrustum.mAspectRatio = header->mAspectRatio;
	mFrustum.mNearPlane = header->mNearPlane;
	mFrustum.mFarPlane = header->mFarPlane;

	mBackgroundColor = header->mBackgroundColor;
}

bool Camera::isCreated() const
{
	return mIsCreated;
}

void Camera::create()
{
	Graphics::create(this, 
					 &mMainFBO, 
					 &mColorTex, 
					 &mDepthTex, 
					 &mColorPickingFBO,
					 &mColorPickingTex,
					 &mColorPickingDepthTex,
					 &mGeometryFBO, 
					 &mPositionTex, 
					 &mNormalTex, 
					 &mSsaoFBO, 
					 &mSsaoColorTex, 
					 &mSsaoNoiseTex, 
					 &mSsaoSamples[0],
					 &mIsCreated);
}

void Camera::destroy()
{
	Graphics::destroy(this,
					  &mMainFBO,
					  &mColorTex,
					  &mDepthTex,
					  &mColorPickingFBO,
					  &mColorPickingTex,
					  &mColorPickingDepthTex,
					  &mGeometryFBO,
				  	  &mPositionTex,
					  &mNormalTex,
					  &mSsaoFBO,
					  &mSsaoColorTex,
					  &mSsaoNoiseTex,
					  &mIsCreated);
}

void Camera::computeViewMatrix(glm::vec3 position, glm::vec3 forward, glm::vec3 up)
{
	viewMatrix = glm::lookAt(position, position + forward, up);
}

void Camera::assignColoring(int color, Guid meshRendererId)
{
	mColoringMap.insert(std::pair<int, Guid>(color, meshRendererId));
}

void Camera::clearColoring()
{
	mColoringMap.clear();
}

glm::mat4 Camera::getViewMatrix() const
{
	return viewMatrix;
}

glm::mat4 Camera::getProjMatrix() const
{
	return glm::perspective(glm::radians(mFrustum.mFov), mFrustum.mAspectRatio, mFrustum.mNearPlane, mFrustum.mFarPlane);
}

glm::vec3 Camera::getSSAOSample(int sample) const
{
	return mSsaoSamples[sample];
}

Guid Camera::getMeshRendererIdAtScreenPos(int x, int y) const
{
	// Note: OpenGL assumes that the window origin is the bottom left corner
	Color color;
	Graphics::readColorPickingPixel(this, x, y, &color);

	int temp = color.r + color.g * 256 + color.b * 256 * 256;

	std::map<int, Guid>::const_iterator it = mColoringMap.find(temp);
	if (it != mColoringMap.end()) {
		return it->second;
	}

	return Guid::INVALID;
}

GLuint Camera::getNativeGraphicsMainFBO() const
{
	return mMainFBO;
}

GLuint Camera::getNativeGraphicsColorPickingFBO() const
{
	return mColorPickingFBO;
}

GLuint Camera::getNativeGraphicsGeometryFBO() const
{
	return mGeometryFBO;
}

GLuint Camera::getNativeGraphicsSSAOFBO() const
{
	return mSsaoFBO;
}

GLuint Camera::getNativeGraphicsColorTex() const
{
	return mColorTex;
}

GLuint Camera::getNativeGraphicsDepthTex() const
{
	return mDepthTex;
}

GLuint Camera::getNativeGraphicsColorPickingTex() const
{
	return mColorPickingTex;
}

GLuint Camera::getNativeGraphicsPositionTex() const
{
	return mPositionTex;
}

GLuint Camera::getNativeGraphicsNormalTex() const
{
	return mNormalTex;
}

GLuint Camera::getNativeGraphicsSSAOColorTex() const
{
	return mSsaoColorTex;
}

GLuint Camera::getNativeGraphicsSSAONoiseTex() const
{
	return mSsaoNoiseTex;
}