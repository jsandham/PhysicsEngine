#include <algorithm>

#include "../../include/core/Serialize.h"
#include "../../include/components/Camera.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Camera::Camera() : Component()
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
    mViewport.mWidth = 1024;
    mViewport.mHeight = 1024;

    mBackgroundColor = glm::vec4(0.15f, 0.15f, 0.15f, 1.0f);

    mFrustum.mFov = 45.0f;
    mFrustum.mAspectRatio = 1.0f;
    mFrustum.mNearPlane = 0.1f;
    mFrustum.mFarPlane = 250.0f;

    mProjMatrix = glm::perspective(glm::radians(mFrustum.mFov), mFrustum.mAspectRatio, mFrustum.mNearPlane,
        mFrustum.mFarPlane);

    mIsCreated = false;
    mIsViewportChanged = false;
}

Camera::Camera(Guid id) : Component(id)
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
    mViewport.mWidth = 1024;
    mViewport.mHeight = 1024;

    mBackgroundColor = glm::vec4(0.15f, 0.15f, 0.15f, 1.0f);

    mFrustum.mFov = 45.0f;
    mFrustum.mAspectRatio = 1.0f;
    mFrustum.mNearPlane = 0.1f;
    mFrustum.mFarPlane = 250.0f;

    mProjMatrix = glm::perspective(glm::radians(mFrustum.mFov), mFrustum.mAspectRatio, mFrustum.mNearPlane,
        mFrustum.mFarPlane);

    mIsCreated = false;
    mIsViewportChanged = false;
}


Camera::~Camera()
{
}

std::vector<char> Camera::serialize() const
{
    return serialize(mId, mEntityId);
}

std::vector<char> Camera::serialize(const Guid &componentId, const Guid &entityId) const
{
    CameraHeader header;
    header.mComponentId = componentId;
    header.mEntityId = entityId;
    header.mTargetTextureId = mTargetTextureId;
    header.mRenderPath = static_cast<uint8_t>(mRenderPath);
    header.mMode = static_cast<uint8_t>(mMode);
    header.mSSAO = static_cast<uint8_t>(mSSAO);
    header.mGizmos = static_cast<uint8_t>(mGizmos);
    header.mBackgroundColor = glm::vec4(mBackgroundColor.r, mBackgroundColor.g, mBackgroundColor.b, mBackgroundColor.a);
    header.mX = static_cast<int32_t>(mViewport.mX);
    header.mY = static_cast<int32_t>(mViewport.mY);
    header.mWidth = static_cast<int32_t>(mViewport.mWidth);
    header.mHeight = static_cast<int32_t>(mViewport.mHeight);
    header.mFov = mFrustum.mFov;
    header.mAspectRatio = mFrustum.mAspectRatio;
    header.mNearPlane = mFrustum.mNearPlane;
    header.mFarPlane = mFrustum.mFarPlane;

    std::vector<char> data(sizeof(CameraHeader));

    memcpy(&data[0], &header, sizeof(CameraHeader));

    return data;
}

void Camera::deserialize(const std::vector<char> &data)
{
    const CameraHeader *header = reinterpret_cast<const CameraHeader *>(&data[0]);

    mId = header->mComponentId;
    mEntityId = header->mEntityId;
    mTargetTextureId = header->mTargetTextureId;

    mRenderPath = static_cast<RenderPath>(header->mRenderPath);
    mMode = static_cast<CameraMode>(header->mMode);
    mSSAO = static_cast<CameraSSAO>(header->mSSAO);
    mGizmos = static_cast<CameraGizmos>(header->mGizmos);

    mViewport.mX = static_cast<int>(header->mX);
    mViewport.mY = static_cast<int>(header->mY);
    mViewport.mWidth = static_cast<int>(header->mWidth);
    mViewport.mHeight = static_cast<int>(header->mHeight);

    mFrustum.mFov = header->mFov;
    mFrustum.mAspectRatio = header->mAspectRatio;
    mFrustum.mNearPlane = header->mNearPlane;
    mFrustum.mFarPlane = header->mFarPlane;

    mProjMatrix = glm::perspective(glm::radians(mFrustum.mFov), mFrustum.mAspectRatio, mFrustum.mNearPlane,
        mFrustum.mFarPlane);

    mBackgroundColor = Color(header->mBackgroundColor);

    mIsViewportChanged = true;
}

void Camera::serialize(std::ostream& out) const
{
    Component::serialize(out);

    PhysicsEngine::write<Guid>(out, mTargetTextureId);
    PhysicsEngine::write<RenderPath>(out, mRenderPath);
    PhysicsEngine::write<CameraMode>(out, mMode);
    PhysicsEngine::write<CameraSSAO>(out, mSSAO);
    PhysicsEngine::write<CameraGizmos>(out, mGizmos);
    PhysicsEngine::write<int>(out, mViewport.mX);
    PhysicsEngine::write<int>(out, mViewport.mY);
    PhysicsEngine::write<int>(out, mViewport.mWidth);
    PhysicsEngine::write<int>(out, mViewport.mHeight);
    PhysicsEngine::write<float>(out, mFrustum.mFov);
    PhysicsEngine::write<float>(out, mFrustum.mAspectRatio);
    PhysicsEngine::write<float>(out, mFrustum.mNearPlane);
    PhysicsEngine::write<float>(out, mFrustum.mFarPlane);
    PhysicsEngine::write<Color>(out, mBackgroundColor);
}

void Camera::deserialize(std::istream& in)
{
    Component::deserialize(in);

    PhysicsEngine::read<Guid>(in, mTargetTextureId);
    PhysicsEngine::read<RenderPath>(in, mRenderPath);
    PhysicsEngine::read<CameraMode>(in, mMode);
    PhysicsEngine::read<CameraSSAO>(in, mSSAO);
    PhysicsEngine::read<CameraGizmos>(in, mGizmos);
    PhysicsEngine::read<int>(in, mViewport.mX);
    PhysicsEngine::read<int>(in, mViewport.mY);
    PhysicsEngine::read<int>(in, mViewport.mWidth);
    PhysicsEngine::read<int>(in, mViewport.mHeight);
    PhysicsEngine::read<float>(in, mFrustum.mFov);
    PhysicsEngine::read<float>(in, mFrustum.mAspectRatio);
    PhysicsEngine::read<float>(in, mFrustum.mNearPlane);
    PhysicsEngine::read<float>(in, mFrustum.mFarPlane);
    PhysicsEngine::read<Color>(in, mBackgroundColor);

    mProjMatrix = glm::perspective(glm::radians(mFrustum.mFov), mFrustum.mAspectRatio, mFrustum.mNearPlane,
        mFrustum.mFarPlane);

    mIsViewportChanged = true;
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
    GLuint64 elapsedTime; // in nanoseconds
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

void Camera::assignColoring(int color, const Guid &transformId)
{
    mColoringMap.insert(std::pair<int, Guid>(color, transformId));
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

    int temp = color.r + color.g * 256 + color.b * 256 * 256;

    std::unordered_map<int, Guid>::const_iterator it = mColoringMap.find(temp);
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

    mProjMatrix = glm::perspective(glm::radians(mFrustum.mFov), mFrustum.mAspectRatio, mFrustum.mNearPlane,
                        mFrustum.mFarPlane);
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