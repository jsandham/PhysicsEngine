#include "../../include/components/Camera.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

float Plane::distance(glm::vec3 point) const
{
	float d = -glm::dot(mN,mX0);

	return (glm::dot(mN,point) + d) / sqrt(glm::dot(mN,mN));
}

float Viewport::getAspectRatio() const
{
	return (float)mHeight / (float)mWidth;
}

int Frustum::checkPoint(glm::vec3 point) const
{
	// loop over all 6 planes
	for(int i = 0; i < 6; i++){
		if(mPlanes[i].distance(point) < 0){
			return -1; // outside
		}
	}

	return 1; // inside
}

int Frustum::checkSphere(glm::vec3 centre, float radius) const
{
	bool answer = 1;

	// loop over all 6 planes
	for(int i = 0; i < 6; i++){
		float distance = mPlanes[i].distance(centre);
		if(distance < -radius){
			return -1; // outside
		}
		else if(distance < radius){
			answer = 0; // intersect
		}
	}

	return answer; // inside
}

int Frustum::checkAABB(glm::vec3 min, glm::vec3 max) const
{
	return 1;
}

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

	mViewport.mX = 0;
	mViewport.mY = 0;
	mViewport.mWidth = 1024;
	mViewport.mHeight = 1024;

	mFrustum.mFov = 45.0f;
	mFrustum.mNearPlane = 0.1f;
	mFrustum.mFarPlane = 250.0f;

	mPosition = glm::vec3(0.0f, 2.0f, 0.0f);
	mFront = glm::vec3(0.0f, 0.0f, -1.0f);
	mUp = glm::vec3(0.0f, 1.0f, 0.0f);
	mBackgroundColor = glm::vec4(0.15f, 0.15f, 0.15f, 1.0f);

	mIsCreated = false;
	mUseSSAO = false;

	updateInternalCameraState();
}

Camera::Camera(std::vector<char> data)
{
	deserialize(data);

	updateInternalCameraState();
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
	header.mPosition = mPosition;
	header.mFront = mFront;
	header.mUp = mUp;
	header.mBackgroundColor = mBackgroundColor;
	header.mX = mViewport.mX;
	header.mY = mViewport.mY;
	header.mWidth = mViewport.mWidth;
	header.mHeight = mViewport.mHeight;
	header.mFov = mFrustum.mFov;
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

	mViewport.mX = header->mX;
	mViewport.mY = header->mY;
	mViewport.mWidth = header->mWidth;
	mViewport.mHeight = header->mHeight;

	mFrustum.mFov = header->mFov;
	mFrustum.mNearPlane = header->mNearPlane;
	mFrustum.mFarPlane = header->mFarPlane;

	mPosition = header->mPosition;
	mFront = header->mFront;
	mUp = header->mUp;
	mBackgroundColor = header->mBackgroundColor;
}

void Camera::updateInternalCameraState()
{
	mFront = glm::normalize(mFront);
	mUp = glm::normalize(mUp);
	mRight = glm::normalize(glm::cross(mFront, mUp));

	float tan = (float)glm::tan(glm::radians(0.5f * mFrustum.mFov));
	float nearPlaneHeight = mFrustum.mNearPlane * tan;
	float nearPlaneWidth = mViewport.getAspectRatio() * nearPlaneHeight;

	// far and near plane intersection along front line
	glm::vec3 fc = mPosition + mFrustum.mFarPlane * mFront;
	glm::vec3 nc = mPosition + mFrustum.mNearPlane * mFront;

	mFrustum.mPlanes[NEAR].mN = mFront;
	mFrustum.mPlanes[NEAR].mX0 = nc;

	mFrustum.mPlanes[FAR].mN = -mFront;
	mFrustum.mPlanes[FAR].mX0 = fc;

	glm::vec3 temp;

	temp = (nc + nearPlaneHeight*mUp) - mPosition;
	temp = glm::normalize(temp);
	mFrustum.mPlanes[TOP].mN = glm::cross(temp, mRight);
	mFrustum.mPlanes[TOP].mX0 = nc + nearPlaneHeight*mUp;

	temp = (nc - nearPlaneHeight*mUp) - mPosition;
	temp = glm::normalize(temp);
	mFrustum.mPlanes[BOTTOM].mN = -glm::cross(temp, mRight);
	mFrustum.mPlanes[BOTTOM].mX0 = nc - nearPlaneHeight*mUp;

	temp = (nc - nearPlaneWidth*mRight) - mPosition;
	temp = glm::normalize(temp);
	mFrustum.mPlanes[LEFT].mN = glm::cross(temp, mUp);
	mFrustum.mPlanes[LEFT].mX0 = nc - nearPlaneWidth*mRight;

	temp = (nc + nearPlaneWidth*mRight) - mPosition;
	temp = glm::normalize(temp);
	mFrustum.mPlanes[RIGHT].mN = -glm::cross(temp, mUp);
	mFrustum.mPlanes[RIGHT].mX0 = nc + nearPlaneWidth*mRight;
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
					  &mGeometryFBO,
				  	  &mPositionTex,
					  &mNormalTex,
					  &mSsaoFBO,
					  &mSsaoColorTex,
					  &mSsaoNoiseTex,
					  &mIsCreated);
}

glm::mat4 Camera::getViewMatrix() const
{
	return glm::lookAt(mPosition, mPosition + mFront, mUp);
}

glm::mat4 Camera::getProjMatrix() const
{
	return glm::perspective(glm::radians(mFrustum.mFov), mViewport.getAspectRatio(), mFrustum.mNearPlane, mFrustum.mFarPlane);
}

glm::vec3 Camera::getSSAOSample(int sample) const
{
	return mSsaoSamples[sample];
}

GLuint Camera::getNativeGraphicsMainFBO() const
{
	return mMainFBO;
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

int Camera::checkPointInFrustum(glm::vec3 point) const
{
	return mFrustum.checkPoint(point);
}

int Camera::checkSphereInFrustum(glm::vec3 centre, float radius) const
{
	return mFrustum.checkSphere(centre, radius);
}

int Camera::checkAABBInFrustum(glm::vec3 min, glm::vec3 max) const
{
	return mFrustum.checkAABB(min, max);
}