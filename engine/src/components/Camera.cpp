#include <algorithm>

#include "../../include/components/Camera.h"
#include "../../include/components/ComponentYaml.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/World.h"

#include "../../include/graphics/Renderer.h"

using namespace PhysicsEngine;

Camera::Camera(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mRenderTextureId = Guid::INVALID;

    mQuery.mQueryBack = 0;
    mQuery.mQueryFront = 1;

    mTargets.mMainFBO = Framebuffer::create(1920, 1080);
    mTargets.mColorPickingFBO = Framebuffer::create(1920, 1080);
    mTargets.mGeometryFBO = Framebuffer::create(1920, 1080, 3, true);
    mTargets.mSsaoFBO = Framebuffer::create(1920, 1080, 1, false);
    mTargets.mOcclusionMapFBO = Framebuffer::create(64, 64, 1, false);

    mRenderPath = RenderPath::Forward;
    mColorTarget = ColorTarget::Color;
    mMode = CameraMode::Main;
    mSSAO = CameraSSAO::SSAO_Off;
    mGizmos = CameraGizmos::Gizmos_Off;
    mShadowCascades = ShadowCascades::FiveCascades;

    mCascadeSplits[0] = 2;
    mCascadeSplits[1] = 4;
    mCascadeSplits[2] = 8;
    mCascadeSplits[3] = 16;
    mCascadeSplits[4] = 100;

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
    mIsViewportChanged = false;
    mRenderToScreen = false;
}

Camera::Camera(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mRenderTextureId = Guid::INVALID;

    mQuery.mQueryBack = 0;
    mQuery.mQueryFront = 1;

    mTargets.mMainFBO = Framebuffer::create(1920, 1080);
    mTargets.mColorPickingFBO = Framebuffer::create(1920, 1080);
    mTargets.mGeometryFBO = Framebuffer::create(1920, 1080, 3, true);
    mTargets.mSsaoFBO = Framebuffer::create(1920, 1080, 1, false);
    mTargets.mOcclusionMapFBO = Framebuffer::create(64, 64, 1, false);

    mRenderPath = RenderPath::Forward;
    mColorTarget = ColorTarget::Color;
    mMode = CameraMode::Main;
    mSSAO = CameraSSAO::SSAO_Off;
    mGizmos = CameraGizmos::Gizmos_Off;
    mShadowCascades = ShadowCascades::FiveCascades;

    mCascadeSplits[0] = 2;
    mCascadeSplits[1] = 4;
    mCascadeSplits[2] = 8;
    mCascadeSplits[3] = 16;
    mCascadeSplits[4] = 100;

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
    mIsViewportChanged = false;
    mRenderToScreen = false;
}

Camera::~Camera()
{
    delete mTargets.mMainFBO;
    delete mTargets.mColorPickingFBO;
    delete mTargets.mGeometryFBO;
    delete mTargets.mSsaoFBO;
    delete mTargets.mOcclusionMapFBO;
}

void Camera::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["entityId"] = mEntityGuid;

    out["renderTextureId"] = mRenderTextureId;
    out["renderPath"] = mRenderPath;
    out["colorTarget"] = mColorTarget;
    out["cameraMode"] = mMode;
    out["cameraSSAO"] = mSSAO;
    out["cameraGizmos"] = mGizmos;
    out["shadowCascades"] = mShadowCascades;
    out["viewport"] = mViewport;
    out["frustum"] = mFrustum;
    out["backgroundColor"] = mBackgroundColor;
    out["cascade splits"] = mCascadeSplits;
    out["enabled"] = mEnabled;
}

void Camera::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

    mEntityGuid = YAML::getValue<Guid>(in, "entityId");

    mRenderTextureId = YAML::getValue<Guid>(in, "renderTextureId");
    mRenderPath = YAML::getValue<RenderPath>(in, "renderPath");
    mColorTarget = YAML::getValue<ColorTarget>(in, "colorTarget");
    mMode = YAML::getValue<CameraMode>(in, "cameraMode");
    mSSAO = YAML::getValue<CameraSSAO>(in, "cameraSSAO");
    mGizmos = YAML::getValue<CameraGizmos>(in, "cameraGizmos");
    mShadowCascades = YAML::getValue<ShadowCascades>(in, "shadowCascades");
    mViewport = YAML::getValue<Viewport>(in, "viewport");
    mFrustum = YAML::getValue<Frustum>(in, "frustum");
    mBackgroundColor = YAML::getValue<Color>(in, "backgroundColor");
    mCascadeSplits = YAML::getValue<std::array<int, 5>>(in, "cascade splits");
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

Guid Camera::getEntityGuid() const
{
    return mEntityGuid;
}

Guid Camera::getGuid() const
{
    return mGuid;
}

Id Camera::getId() const
{
    return mId;
}

bool Camera::isViewportChanged() const
{
    return mIsViewportChanged;
}

void Camera::resizeTargets()
{
}

void Camera::beginQuery()
{
    mQuery.mNumInstancedDrawCalls = 0;
    mQuery.mNumDrawCalls = 0;
    mQuery.mTotalElapsedTime = 0.0f;
    mQuery.mVerts = 0;
    mQuery.mTris = 0;
    mQuery.mLines = 0;
    mQuery.mPoints = 0;

    // Renderer::getRenderer()->beginQuery(mQuery.mQueryId[mQuery.mQueryBack]);
}

void Camera::endQuery()
{
    unsigned long long elapsedTime = 0; // in nanoseconds

    // Renderer::getRenderer()->endQuery(mQuery.mQueryId[mQuery.mQueryFront], &elapsedTime);

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

void Camera::computeViewMatrix(const glm::vec3 &position, const glm::vec3 &forward, const glm::vec3 &up,
                               const glm::vec3 &right)
{
    mPosition = position;
    mForward = forward;
    mUp = up;
    mRight = right;
    mViewMatrix = glm::lookAt(position, position + forward, up);
    mInvViewMatrix = glm::inverse(mViewMatrix);

    // update frustum planes
    mFrustum.computePlanes(mPosition, mForward, mUp, mRight);
}

void Camera::setColoringIds(const std::vector<Id> &ids)
{
    mColoringIds = ids;
}

glm::vec3 Camera::getPosition() const
{
    return mPosition;
}

glm::vec3 Camera::getForward() const
{
    return mForward;
}

glm::vec3 Camera::getUp() const
{
    return mUp;
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

Id Camera::getTransformIdAtScreenPos(int x, int y) const
{
    // Note: OpenGL assumes that the window origin is the bottom left corner
    Color32 color;
    mTargets.mColorPickingFBO->readColorAtPixel(x, y, &color);

    uint32_t i = Color32::convertColor32ToUint32(color);
    if ((i - 1) < mColoringIds.size() && i >= 1)
    {
        return mColoringIds[i - 1];
    }

    return Id::INVALID;
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

    // update frustum planes
    mFrustum.computePlanes(mPosition, mForward, mUp, mRight);
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

std::array<int, 5> Camera::getCascadeSplits() const
{
    return mCascadeSplits;
}

std::array<float, 6> Camera::calcViewSpaceCascadeEnds() const
{
    float nearDist = mFrustum.mNearPlane;
    float farDist = mFrustum.mFarPlane;

    std::array<float, 6> cascadeEnds;

    cascadeEnds[0] = -1 * nearDist;
    cascadeEnds[1] = -1 * (nearDist + (farDist - nearDist) * (mCascadeSplits[0] / 100.0f));
    cascadeEnds[2] = -1 * (nearDist + (farDist - nearDist) * (mCascadeSplits[1] / 100.0f));
    cascadeEnds[3] = -1 * (nearDist + (farDist - nearDist) * (mCascadeSplits[2] / 100.0f));
    cascadeEnds[4] = -1 * (nearDist + (farDist - nearDist) * (mCascadeSplits[3] / 100.0f));
    cascadeEnds[5] = -1 * (nearDist + (farDist - nearDist) * (mCascadeSplits[4] / 100.0f));

    return cascadeEnds;
}

std::array<Frustum, 5> Camera::calcCascadeFrustums(const std::array<float, 6> &cascadeEnds) const
{
    float fov = mFrustum.mFov;
    float aspect = mFrustum.mAspectRatio;

    std::array<Frustum, 5> frustums;
    for (size_t i = 0; i < frustums.size(); i++)
    {
        frustums[i] = Frustum(fov, aspect, -cascadeEnds[i], -cascadeEnds[i + 1]);
    }

    return frustums;
}

void Camera::setCascadeSplit(size_t splitIndex, int splitValue)
{
    if (splitIndex < 0 || splitIndex >= mCascadeSplits.size())
    {
        return;
    }

    mCascadeSplits[splitIndex] = splitValue;

    mCascadeSplits[0] = std::min(std::max(mCascadeSplits[0], 0), 100);
    for (size_t i = 1; i < mCascadeSplits.size(); i++)
    {
        mCascadeSplits[i] = std::min(std::max(mCascadeSplits[i], mCascadeSplits[i - 1]), 100);
    }
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

Framebuffer *Camera::getNativeGraphicsMainFBO() const
{
    return mTargets.mMainFBO;
}

Framebuffer *Camera::getNativeGraphicsColorPickingFBO() const
{
    return mTargets.mColorPickingFBO;
}

Framebuffer *Camera::getNativeGraphicsGeometryFBO() const
{
    return mTargets.mGeometryFBO;
}

Framebuffer *Camera::getNativeGraphicsSSAOFBO() const
{
    return mTargets.mSsaoFBO;
}

Framebuffer *Camera::getNativeGraphicsOcclusionMapFBO() const
{
    return mTargets.mOcclusionMapFBO;
}

RenderTextureHandle *Camera::getNativeGraphicsColorTex() const
{
    return mTargets.mMainFBO->getColorTex();
}

RenderTextureHandle *Camera::getNativeGraphicsDepthTex() const
{
    return mTargets.mMainFBO->getDepthTex();
}

RenderTextureHandle *Camera::getNativeGraphicsColorPickingTex() const
{
    return mTargets.mColorPickingFBO->getColorTex();
}

RenderTextureHandle *Camera::getNativeGraphicsPositionTex() const
{
    return mTargets.mGeometryFBO->getColorTex();
}

RenderTextureHandle *Camera::getNativeGraphicsNormalTex() const
{
    return mTargets.mGeometryFBO->getColorTex();
}

RenderTextureHandle *Camera::getNativeGraphicsAlbedoSpecTex() const
{
    return mTargets.mGeometryFBO->getColorTex();
}

RenderTextureHandle *Camera::getNativeGraphicsSSAOColorTex() const
{
    return mTargets.mSsaoFBO->getColorTex();
}

RenderTextureHandle *Camera::getNativeGraphicsSSAONoiseTex() const
{
    return mTargets.mSsaoFBO->getColorTex();
}

RenderTextureHandle *Camera::getNativeGraphicsOcclusionMapTex() const
{
    return mTargets.mOcclusionMapFBO->getColorTex();
}

Entity *Camera::getEntity() const
{
    return mWorld->getActiveScene()->getEntityByGuid(mEntityGuid);
}