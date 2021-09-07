#include <algorithm>

#include "../../include/components/Camera.h"
//#include "../../include/core/Serialization.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Camera::Camera(World* world) : Component(world)
{
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
    mGizmos = CameraGizmos::Gizmos_Off;

    mViewport.mX = 0;
    mViewport.mY = 0;
    mViewport.mWidth = 1920;
    mViewport.mHeight = 1080;

    mBackgroundColor = glm::vec4(0.15f, 0.15f, 0.15f, 1.0f);

    mFrustum.mFov = 45.0f;
    mFrustum.mAspectRatio = 1.0f;
    mFrustum.mNearPlane = 0.1f;
    mFrustum.mFarPlane = 250.0f;

    mProjMatrix =
        glm::perspective(glm::radians(mFrustum.mFov), mFrustum.mAspectRatio, mFrustum.mNearPlane, mFrustum.mFarPlane);

    mEnabled = true;
    mIsCreated = false;
    mIsViewportChanged = false;
}

Camera::Camera(World* world, Guid id) : Component(world, id)
{
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
    mGizmos = CameraGizmos::Gizmos_Off;

    mViewport.mX = 0;
    mViewport.mY = 0;
    mViewport.mWidth = 1920;
    mViewport.mHeight = 1080;

    mBackgroundColor = glm::vec4(0.15f, 0.15f, 0.15f, 1.0f);

    mFrustum.mFov = 45.0f;
    mFrustum.mAspectRatio = 1.0f;
    mFrustum.mNearPlane = 0.1f;
    mFrustum.mFarPlane = 250.0f;

    mProjMatrix =
        glm::perspective(glm::radians(mFrustum.mFov), mFrustum.mAspectRatio, mFrustum.mNearPlane, mFrustum.mFarPlane);

    mEnabled = true;
    mIsCreated = false;
    mIsViewportChanged = false;
}

Camera::~Camera()
{
}

void Camera::serialize(YAML::Node &out) const
{
    Component::serialize(out);

    out["targetTextureId"] = mTargetTextureId;
    out["renderPath"] = mRenderPath;
    out["cameraMode"] = mMode;
    out["cameraSSAO"] = mSSAO;
    out["cameraGizmos"] = mGizmos;
    out["viewport"] = mViewport;
    out["frustum"] = mFrustum;
    out["backgroundColor"] = mBackgroundColor;
    out["enabled"] = mEnabled;
}

void Camera::deserialize(const YAML::Node &in)
{
    Component::deserialize(in);

    mTargetTextureId = YAML::getValue<Guid>(in, "targetTextureId");
    mRenderPath = YAML::getValue<RenderPath>(in, "renderPath");
    mMode = YAML::getValue<CameraMode>(in, "cameraMode");
    mSSAO = YAML::getValue<CameraSSAO>(in, "cameraSSAO");
    mGizmos = YAML::getValue<CameraGizmos>(in, "cameraGizmos");
    mViewport = YAML::getValue<Viewport>(in, "viewport");
    mFrustum = YAML::getValue<Frustum>(in, "frustum");
    mBackgroundColor = YAML::getValue<Color>(in, "backgroundColor");
    mEnabled = YAML::getValue<bool>(in, "enabled");

    mProjMatrix =
        glm::perspective(glm::radians(mFrustum.mFov), mFrustum.mAspectRatio, mFrustum.mNearPlane, mFrustum.mFarPlane);

    mIsViewportChanged = true;
}

int Camera::getType() const
{
    return PhysicsEngine::CAMERA_TYPE;
}

std::string Camera::getObjectName() const
{
    return PhysicsEngine::CAMERA_NAME;
}

bool Camera::isCreated() const
{
    return mIsCreated;
}

bool Camera::isViewportChanged() const
{
    return mIsViewportChanged;
}

void Camera::createTargets()
{
    Graphics::createTargets(&mTargets, mViewport, &mSsaoSamples[0], &mQuery.mQueryId[0], &mQuery.mQueryId[1]);

    mIsCreated = true;
}

void Camera::destroyTargets()
{
    Graphics::destroyTargets(&mTargets, &mQuery.mQueryId[0], &mQuery.mQueryId[1]);

    mIsCreated = false;
}

void Camera::resizeTargets()
{
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
}

void Camera::endQuery()
{
    unsigned long long elapsedTime; // in nanoseconds
    Graphics::endQuery(mQuery.mQueryId[mQuery.mQueryFront], &elapsedTime);

    mQuery.mTotalElapsedTime += elapsedTime / 1000000.0f;

    // swap which query is active
    if (mQuery.mQueryBack)
    {
        mQuery.mQueryBack = 0;
        mQuery.mQueryFront = 1;
    }
    else
    {
        mQuery.mQueryBack = 1;
        mQuery.mQueryFront = 0;
    }
}

void Camera::computeViewMatrix(const glm::vec3 &position, const glm::vec3 &forward, const glm::vec3 &up)
{
    mPosition = position;
    mViewMatrix = glm::lookAt(position, position + forward, up);
    mInvViewMatrix = glm::inverse(mViewMatrix);
}

void Camera::assignColoring(Color32 color, const Guid& transformId)
{
    mColoringMap.insert(std::pair<Color32, Guid>(color, transformId));
}

void Camera::clearColoring()
{
    mColoringMap.clear();
}

glm::vec3 Camera::getPosition() const
{
    return mPosition;
}

glm::mat4 Camera::getViewMatrix() const
{
    return mViewMatrix;
}

glm::mat4 Camera::getInvViewMatrix() const
{
    return mInvViewMatrix;
}

glm::mat4 Camera::getProjMatrix() const
{
    return mProjMatrix;
}

glm::vec3 Camera::getSSAOSample(int sample) const
{
    return mSsaoSamples[sample];
}

Guid Camera::getTransformIdAtScreenPos(int x, int y) const
{
    // Note: OpenGL assumes that the window origin is the bottom left corner
    Color32 color;
    Graphics::readColorPickingPixel(&mTargets, x, y, &color);

    std::unordered_map<Color32, Guid>::const_iterator it = mColoringMap.find(color);
    if (it != mColoringMap.end())
    {
        return it->second;
    }

    return Guid::INVALID;
}

Frustum Camera::getFrustum() const
{
    return mFrustum;
}

Viewport Camera::getViewport() const
{
    return mViewport;
}

void Camera::setFrustum(float fov, float aspectRatio, float nearPlane, float farPlane)
{
    mFrustum.mFov = fov;
    mFrustum.mAspectRatio = aspectRatio;
    mFrustum.mNearPlane = nearPlane;
    mFrustum.mFarPlane = farPlane;

    mProjMatrix =
        glm::perspective(glm::radians(mFrustum.mFov), mFrustum.mAspectRatio, mFrustum.mNearPlane, mFrustum.mFarPlane);
}

void Camera::setViewport(int x, int y, int width, int height)
{
    mIsViewportChanged =
        mViewport.mX != x || mViewport.mY != y || mViewport.mWidth != width || mViewport.mHeight != height;

    mViewport.mX = x;
    mViewport.mY = y;
    mViewport.mWidth = width;
    mViewport.mHeight = height;
}

Ray Camera::normalizedDeviceSpaceToRay(float x, float y) const
{
    // compute ray cast from the normalized device coordinate ([-1, 1] x [-1, 1]) into the scene
    // gives mouse pixel coordinates in the [-1, 1] range
    x = std::min(1.0f, std::max(-1.0f, x));
    y = std::min(1.0f, std::max(-1.0f, y));

    glm::vec4 start = glm::vec4(x, y, 0, 1.0f);
    glm::vec4 end = glm::vec4(x, y, 1.0f, 1.0f);

    glm::mat4 invProjMatrix = glm::inverse(getProjMatrix());
    glm::mat4 invViewMatrix = getInvViewMatrix();

    // transform to view space
    start = invProjMatrix * start;
    end = invProjMatrix * end;

    // transform to world space
    start = invViewMatrix * start;
    end = invViewMatrix * end;

    start *= 1.0f / start.w;
    end *= 1.0f / end.w;

    glm::vec4 direction = glm::normalize(end - start);

    Ray ray;
    ray.mOrigin = glm::vec3(start.x, start.y, start.z);
    ray.mDirection = glm::vec3(direction.x, direction.y, direction.z);

    return ray;
}

Ray Camera::screenSpaceToRay(int x, int y) const
{
    // compute ray cast from the screen space ([0, 0] x [pixelWidth, pixelHeight]) into the scene
    x = std::min(mViewport.mWidth, std::max(0, x));
    y = std::min(mViewport.mHeight, std::max(0, y));

    float ndcX = (2.0f * x - mViewport.mWidth) / mViewport.mWidth;
    float ndcY = (2.0f * y - mViewport.mHeight) / mViewport.mHeight;

    return normalizedDeviceSpaceToRay(ndcX, ndcY);
}

unsigned int Camera::getNativeGraphicsMainFBO() const
{
    return mTargets.mMainFBO;
}

unsigned int Camera::getNativeGraphicsColorPickingFBO() const
{
    return mTargets.mColorPickingFBO;
}

unsigned int Camera::getNativeGraphicsGeometryFBO() const
{
    return mTargets.mGeometryFBO;
}

unsigned int Camera::getNativeGraphicsSSAOFBO() const
{
    return mTargets.mSsaoFBO;
}

unsigned int Camera::getNativeGraphicsColorTex() const
{
    return mTargets.mColorTex;
}

unsigned int Camera::getNativeGraphicsDepthTex() const
{
    return mTargets.mDepthTex;
}

unsigned int Camera::getNativeGraphicsColorPickingTex() const
{
    return mTargets.mColorPickingTex;
}

unsigned int Camera::getNativeGraphicsPositionTex() const
{
    return mTargets.mPositionTex;
}

unsigned int Camera::getNativeGraphicsNormalTex() const
{
    return mTargets.mNormalTex;
}

unsigned int Camera::getNativeGraphicsAlbedoSpecTex() const
{
    return mTargets.mAlbedoSpecTex;
}

unsigned int Camera::getNativeGraphicsSSAOColorTex() const
{
    return mTargets.mSsaoColorTex;
}

unsigned int Camera::getNativeGraphicsSSAONoiseTex() const
{
    return mTargets.mSsaoNoiseTex;
}