#include "../../include/components/Camera.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Camera::Camera()
{
	mComponentId = Guid::INVALID;
	mEntityId = Guid::INVALID;
	mTargetTextureId = Guid::INVALID;

	mQuery.mQueryBack = 0;
	mQuery.mQueryFront = 1;

	mTargets.mMainFBO = 0;
	mTargets.mColorTex = 0;
	mTargets.mDepthTex = 0;

	mTargets.mColorPickingFBO = 0;
	mTargets.mColorPickingTex = 0;
	mTargets.mColorPickingDepthTex = 0;

	mTargets.mGeometryFBO = 0;
	mTargets.mPositionTex = 0;
	mTargets.mNormalTex = 0;
	mTargets.mAlbedoSpecTex = 0;

	mTargets.mSsaoFBO = 0;
	mTargets.mSsaoColorTex = 0;
	mTargets.mSsaoNoiseTex = 0;

	mRenderPath = RenderPath::Forward;
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

Camera::Camera(const std::vector<char>& data)
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
	header.mRenderPath = mRenderPath;
	header.mMode = mMode;
	header.mSSAO = mSSAO;
	header.mBackgroundColor = glm::vec4(mBackgroundColor.r,
										mBackgroundColor.g,
										mBackgroundColor.b, 
										mBackgroundColor.a);
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

void Camera::deserialize(const std::vector<char>& data)
{
	const CameraHeader* header = reinterpret_cast<const CameraHeader*>(&data[0]);

	mComponentId = header->mComponentId;
	mEntityId = header->mEntityId;
	mTargetTextureId = header->mTargetTextureId;

	mRenderPath = header->mRenderPath;
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

	mBackgroundColor = Color(header->mBackgroundColor);
}

bool Camera::isCreated() const
{
	return mIsCreated;
}

void Camera::create()
{
	Graphics::create(this, 
					 &mTargets.mMainFBO, 
					 &mTargets.mColorTex,
					 &mTargets.mDepthTex,
					 &mTargets.mColorPickingFBO,
					 &mTargets.mColorPickingTex,
					 &mTargets.mColorPickingDepthTex,
					 &mTargets.mGeometryFBO,
					 &mTargets.mPositionTex,
					 &mTargets.mNormalTex,
					 &mTargets.mAlbedoSpecTex,
					 &mTargets.mSsaoFBO,
					 &mTargets.mSsaoColorTex,
					 &mTargets.mSsaoNoiseTex,
					 &mSsaoSamples[0],
					 &mQuery.mQueryId[0],
					 &mQuery.mQueryId[1],
					 &mIsCreated);
}

void Camera::destroy()
{
	Graphics::destroy(this,
					  &mTargets.mMainFBO,
					  &mTargets.mColorTex,
					  &mTargets.mDepthTex,
					  &mTargets.mColorPickingFBO,
					  &mTargets.mColorPickingTex,
					  &mTargets.mColorPickingDepthTex,
					  &mTargets.mGeometryFBO,
				  	  &mTargets.mPositionTex,
					  &mTargets.mNormalTex,
					  &mTargets.mAlbedoSpecTex,
					  &mTargets.mSsaoFBO,
					  &mTargets.mSsaoColorTex,
					  &mTargets.mSsaoNoiseTex,
					  &mQuery.mQueryId[0],
					  &mQuery.mQueryId[1],
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

void Camera::beginQuery()
{
	mQuery.mNumBatchDrawCalls = 0;
	mQuery.mNumDrawCalls = 0;
	mQuery.mTotalElapsedTime = 0.0f;
	mQuery.mVerts = 0;
	mQuery.mTris = 0;
	mQuery.mLines = 0;
	mQuery.mPoints = 0;

	Graphics::beginQuery(mQuery.mQueryId[mQuery.mQueryBack]);

	//glBeginQuery(GL_TIME_ELAPSED, mQuery.mQueryId[mQuery.mQueryBack]);
}

void Camera::endQuery()
{
	//glEndQuery(GL_TIME_ELAPSED);

	//GLuint64 elapsedTime; // in nanoseconds
	//glGetQueryObjectui64v(mQuery.mQueryId[mQuery.mQueryFront], GL_QUERY_RESULT, &elapsedTime);

	GLuint64 elapsedTime; // in nanoseconds
	Graphics::endQuery(mQuery.mQueryId[mQuery.mQueryFront], &elapsedTime);

	mQuery.mTotalElapsedTime += elapsedTime / 1000000.0f;

	// swap which query is active
	if (mQuery.mQueryBack) {
		mQuery.mQueryBack = 0;
		mQuery.mQueryFront = 1;
	}
	else {
		mQuery.mQueryBack = 1;
		mQuery.mQueryFront = 0;
	}
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
	Color32 color;
	Graphics::readColorPickingPixel(this, x, y, &color);

	int temp = color.r + color.g * 256 + color.b * 256 * 256;

	std::unordered_map<int, Guid>::const_iterator it = mColoringMap.find(temp);
	if (it != mColoringMap.end()) {
		return it->second;
	}

	return Guid::INVALID;
}

GLuint Camera::getNativeGraphicsMainFBO() const
{
	return mTargets.mMainFBO;
}

GLuint Camera::getNativeGraphicsColorPickingFBO() const
{
	return mTargets.mColorPickingFBO;
}

GLuint Camera::getNativeGraphicsGeometryFBO() const
{
	return mTargets.mGeometryFBO;
}

GLuint Camera::getNativeGraphicsSSAOFBO() const
{
	return mTargets.mSsaoFBO;
}

GLuint Camera::getNativeGraphicsColorTex() const
{
	return mTargets.mColorTex;
}

GLuint Camera::getNativeGraphicsDepthTex() const
{
	return mTargets.mDepthTex;
}

GLuint Camera::getNativeGraphicsColorPickingTex() const
{
	return mTargets.mColorPickingTex;
}

GLuint Camera::getNativeGraphicsPositionTex() const
{
	return mTargets.mPositionTex;
}

GLuint Camera::getNativeGraphicsNormalTex() const
{
	return mTargets.mNormalTex;
}

GLuint Camera::getNativeGraphicsAlbedoSpecTex() const
{
	return mTargets.mAlbedoSpecTex;
}

GLuint Camera::getNativeGraphicsSSAOColorTex() const
{
	return mTargets.mSsaoColorTex;
}

GLuint Camera::getNativeGraphicsSSAONoiseTex() const
{
	return mTargets.mSsaoNoiseTex;
}