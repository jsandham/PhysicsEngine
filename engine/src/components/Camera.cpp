#include <algorithm>
#include <random>
#include <omp.h>

#include "../../include/components/Camera.h"
#include "../../include/components/ComponentYaml.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/AssetEnums.h"
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
    mTargets.mRaytracingTex = RenderTextureHandle::create(
        256, 256, TextureFormat::RGB, TextureWrapMode::ClampToBorder, TextureFilterMode::Nearest);

    mRenderPath = RenderPath::Forward;
    mColorTarget = ColorTarget::Color;
    mMode = CameraMode::Main;
    mSSAO = CameraSSAO::SSAO_Off;
    mShadowCascades = ShadowCascades::FiveCascades;

    mGizmos.mShowFrustums = false;
    mGizmos.mShowLights = false;
    mGizmos.mShowBVH = false;
    mGizmos.mShowBoundingSheres = false;
    mGizmos.mShowBoundingAABBs = false;
    mGizmos.mShowGrid = false;
    mGizmos.mTurnOnSphereIntersectDemo = false;
    mGizmos.mTurnOnAABBIntersectionDemo = false;

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

    mProjMatrix = glm::perspective(glm::radians(mFrustum.mFov), mFrustum.mAspectRatio, mFrustum.mNearPlane, mFrustum.mFarPlane);
    mInvProjMatrix = glm::inverse(mProjMatrix);

    mEnabled = true;
    mIsViewportChanged = false;
    mMoved = false;
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
    mTargets.mRaytracingTex = RenderTextureHandle::create(256, 256, TextureFormat::RGB, TextureWrapMode::ClampToBorder,
                                                          TextureFilterMode::Nearest);

    mRenderPath = RenderPath::Forward;
    mColorTarget = ColorTarget::Color;
    mMode = CameraMode::Main;
    mSSAO = CameraSSAO::SSAO_Off;
    mShadowCascades = ShadowCascades::FiveCascades;

    mGizmos.mShowFrustums = false;
    mGizmos.mShowLights = false;
    mGizmos.mShowBVH = false;
    mGizmos.mShowBoundingSheres = false;
    mGizmos.mShowBoundingAABBs = false;
    mGizmos.mShowGrid = false;
    mGizmos.mTurnOnSphereIntersectDemo = false;
    mGizmos.mTurnOnAABBIntersectionDemo = false;

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

    mProjMatrix = glm::perspective(glm::radians(mFrustum.mFov), mFrustum.mAspectRatio, mFrustum.mNearPlane,
                                                mFrustum.mFarPlane);
    mInvProjMatrix = glm::inverse(mProjMatrix);

    mEnabled = true;
    mIsViewportChanged = false;
    mMoved = false;
    mRenderToScreen = false;
}

Camera::~Camera()
{
    delete mTargets.mMainFBO;
    delete mTargets.mColorPickingFBO;
    delete mTargets.mGeometryFBO;
    delete mTargets.mSsaoFBO;
    delete mTargets.mOcclusionMapFBO;
    delete mTargets.mRaytracingTex;
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
    mShadowCascades = YAML::getValue<ShadowCascades>(in, "shadowCascades");
    mViewport = YAML::getValue<Viewport>(in, "viewport");
    mFrustum = YAML::getValue<Frustum>(in, "frustum");
    mBackgroundColor = YAML::getValue<Color>(in, "backgroundColor");
    mCascadeSplits = YAML::getValue<std::array<int, 5>>(in, "cascade splits");
    mEnabled = YAML::getValue<bool>(in, "enabled");

    mProjMatrix = glm::perspective(glm::radians(mFrustum.mFov), mFrustum.mAspectRatio, mFrustum.mNearPlane, mFrustum.mFarPlane);
    mInvProjMatrix = glm::inverse(mProjMatrix);

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

bool Camera::moved() const
{
    return mMoved;
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
    mMoved = false;
    if (mPosition != position || mForward != forward || mUp != up || mRight != right)
    {
        mMoved = true;
    }

    mPosition = position;
    mForward = forward;
    mUp = up;
    mRight = right;

    mViewMatrix = glm::lookAt(position, position + forward, up);
    mViewProjMatrix = mProjMatrix * mViewMatrix;
    mInvViewMatrix = glm::inverse(mViewMatrix);
    mInvViewProjMatrix = glm::inverse(mViewProjMatrix);

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

glm::vec3 Camera::getRight() const
{
    return mRight;
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

glm::mat4 Camera::getInvProjMatrix() const
{
    return mInvProjMatrix;
}

glm::mat4 Camera::getViewProjMatrix() const
{
    return mViewProjMatrix;
}

glm::mat4 Camera::getInvViewProjMatrix() const
{
    return mInvViewProjMatrix;
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

    mProjMatrix = glm::perspective(glm::radians(mFrustum.mFov), mFrustum.mAspectRatio, mFrustum.mNearPlane, mFrustum.mFarPlane);
    mInvProjMatrix = glm::inverse(mProjMatrix);

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

RenderTextureHandle *Camera::getNativeGraphicsRaytracingTex() const
{
    return mTargets.mRaytracingTex;
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

// Iterative computeColor for spheres
static glm::vec3 computeColorIterative(const BVH &bvh, const std::vector<Sphere> &spheres,
                                       const std::vector<RaytraceMaterial> &materials, const Ray &ray, int maxDepth)
{
    Ray ray2 = ray;

    glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f);

    for (int depth = 0; depth < maxDepth; depth++)
    {
        BVHHit hit = bvh.intersect(ray2, spheres);

        if (hit.mIndex >= 0)
        {
            if (materials[hit.mIndex].mType == RaytraceMaterial::MaterialType::DiffuseLight)
            {
                color *= materials[hit.mIndex].mEmissive;
                break;
            }

            glm::vec3 point = ray2.getPoint(hit.mT);
            glm::vec3 normal = spheres[hit.mIndex].getNormal(point);

            switch (materials[hit.mIndex].mType)
            {
            case RaytraceMaterial::MaterialType::Lambertian:
                ray2 = RaytraceMaterial::generate_lambertian_ray(point, normal);
                break;
            case RaytraceMaterial::MaterialType::Metallic:
                ray2 = RaytraceMaterial::generate_metallic_ray(point, ray.mDirection, normal,
                                                               materials[hit.mIndex].mFuzz);
                break;
            case RaytraceMaterial::MaterialType::Dialectric:
                ray2 = RaytraceMaterial::generate_dialectric_ray(point, ray.mDirection, normal,
                                                                 materials[hit.mIndex].mRefractionIndex);
                break;
            }

            color *= materials[hit.mIndex].mAlbedo;
        }
        else
        {
            // color *= glm::vec3(0.0f, 0.0f, 0.0f);
            glm::vec3 unit_direction = glm::normalize(ray2.mDirection);
            float a = 0.5f * (unit_direction.y + 1.0f);
            color *= (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
            break;
        }
    }

    return color;
}

// Iterative computeColor using TLAS and BLAS
static glm::vec3 computeColorIterative(const TLAS &tlas, const std::vector<BLAS> &blas,
                                       const std::vector<RaytraceMaterial> &materials, const Ray &ray, int maxDepth)
{
    Ray ray2 = ray;

    glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f);

    for (int depth = 0; depth < maxDepth; depth++)
    {
        TLASHit hit = tlas.intersectTLAS(ray2);

        if (hit.blasIndex >= 0)
        {
            if (materials[hit.blasIndex].mType == RaytraceMaterial::MaterialType::DiffuseLight)
            {
                color *= materials[hit.blasIndex].mEmissive;
                break;
            }

            glm::vec3 normal = glm::normalize(blas[hit.blasIndex].getTriangle(hit.mTriIndex).getNormal());
            glm::vec3 point = ray2.getPoint(hit.mT);

            switch (materials[hit.blasIndex].mType)
            {
            case RaytraceMaterial::MaterialType::Lambertian:
                ray2 = RaytraceMaterial::generate_lambertian_ray(point, normal);
                break;
            case RaytraceMaterial::MaterialType::Metallic:
                ray2 = RaytraceMaterial::generate_metallic_ray(point, ray.mDirection, normal,
                                                               materials[hit.blasIndex].mFuzz);
                break;
            case RaytraceMaterial::MaterialType::Dialectric:
                ray2 = RaytraceMaterial::generate_dialectric_ray(point, ray.mDirection, normal,
                                                                 materials[hit.blasIndex].mRefractionIndex);
                break;
            }

            color *= materials[hit.blasIndex].mAlbedo;
        }
        else
        {
            color *= glm::vec3(0.01f, 0.01f, 0.01f);
            //glm::vec3 unit_direction = glm::normalize(ray2.mDirection);
            //float a = 0.5f * (unit_direction.y + 1.0f);
            //color *= (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
            break;
        }
    }

    return color;
}

static uint32_t pcg_hash(uint32_t seed)
{
    uint32_t state = seed * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static float generate_rand(float a = 0.0f, float b = 1.0f)
{
    static uint32_t seed = 1234567;
    seed++;

    float uniform = (float)pcg_hash(seed) / (float)std::numeric_limits<uint32_t>::max();
    return a + (b - a) * uniform;
}

static glm::vec2 generatePixelSampleNDC(int u, int v, float du, float dv)
{
    // NDC coordinates for 2x2x2 cube [-1, 1]x[-1, 1]x[-1, 1]
    //         +y |   * +z
    //            |  *
    //            | *
    // -x ________|*________ +x
    // In frustum plane, bottom, left, far corner correspnds to NDC point [-1, -1, 1]
    glm::vec2 bottomLeft_NDC = glm::vec2(-1.0f, -1.0f);

    // Bottom, left corner pixel centre in NDC
    glm::vec2 pixelCentreBottomLeft_NDC = bottomLeft_NDC + glm::vec2(0.5f * du, 0.5f * dv);

    // Plane pixel centre in NDC
    glm::vec2 pixelCentre_NDC = pixelCentreBottomLeft_NDC + glm::vec2(u * du, v * dv);

    // Randomly sample from pixel
    return pixelCentre_NDC + glm::vec2(generate_rand(-0.5f, 0.5f) * du, generate_rand(-0.5f, 0.5f) * dv);
}

void Camera::clearPixels()
{
    for (size_t i = 0; i < mImage.size(); i++)
    {
        mImage[i] = 0.0f;
    }
    for (size_t i = 0; i < mSamplesPerRay.size(); i++)
    {
        mSamplesPerRay[i] = 0;
    }
}

void Camera::resizePixels()
{
    // Image size
    int width = getNativeGraphicsRaytracingTex()->getWidth();
    int height = getNativeGraphicsRaytracingTex()->getHeight();

    if (width * height * 3 != mImage.size())
    {
        mImage.resize(width * height * 3);
        mSamplesPerRay.resize(width * height);
        for (size_t i = 0; i < mImage.size(); i++)
        {
            mImage[i] = 0.0f;
        }

        for (size_t i = 0; i < mSamplesPerRay.size(); i++)
        {
            mSamplesPerRay[i] = 0;
        }
    }
}

void Camera::raytraceSpheres(const BVH &bvh, const std::vector<Sphere> &spheres,
                           const std::vector<RaytraceMaterial> &materials, int maxBounces, int maxSamples)
{
    // Image size
    int width = getNativeGraphicsRaytracingTex()->getWidth();
    int height = getNativeGraphicsRaytracingTex()->getHeight();

    // In NDC we use a 2x2x2 box ranging from [-1,1]x[-1,1]x[-1,1]
    float du = 2.0f / 256;
    float dv = 2.0f / 256;

    constexpr int TILE_WIDTH = 8;
    constexpr int TILE_HEIGHT = 8;

    constexpr int TILE_ROWS = 256 / TILE_HEIGHT;
    constexpr int TILE_COLUMNS = 256 / TILE_WIDTH;

    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < TILE_ROWS * TILE_COLUMNS; t++)
    {
        int brow = t / TILE_ROWS;
        int bcol = t % TILE_COLUMNS;

        for (int r = 0; r < TILE_HEIGHT; r++)
        {
            for (int c = 0; c < TILE_WIDTH; c++)
            {
                int row = (TILE_HEIGHT * brow + r);
                int col = (TILE_WIDTH * bcol + c);

                glm::vec2 pixelSampleNDC = generatePixelSampleNDC(col, row, du, dv);

                int irow = (int)(height * (0.5f * (pixelSampleNDC.y + 1.0f)));
                int icol = (int)(width * (0.5f * (pixelSampleNDC.x + 1.0f)));

                irow = glm::min(height - 1, glm::max(0, irow));
                icol = glm::min(width - 1, glm::max(0, icol));

                int offset = width * irow + icol;

                assert(offset >= 0);
                assert(offset < width * height);

                mSamplesPerRay[offset]++;

                // Read color from image
                float red = mImage[3 * offset + 0];
                float green = mImage[3 * offset + 1];
                float blue = mImage[3 * offset + 2];
                glm::vec3 color = glm::vec3(red, green, blue);

                color += computeColorIterative(bvh, spheres, materials, getCameraRay(pixelSampleNDC), maxBounces);
                
                // Store computed color to image
                mImage[3 * offset + 0] = color.r;
                mImage[3 * offset + 1] = color.g;
                mImage[3 * offset + 2] = color.b;
            }
        }
    }
}

void Camera::raytraceScene(const TLAS &tlas, const std::vector<BLAS> &blas, const std::vector<RaytraceMaterial> &materials, int maxBounces, int maxSamples)
{
    // Image size
    int width = getNativeGraphicsRaytracingTex()->getWidth();
    int height = getNativeGraphicsRaytracingTex()->getHeight();

    // In NDC we use a 2x2x2 box ranging from [-1,1]x[-1,1]x[-1,1]
    float du = 2.0f / 256;
    float dv = 2.0f / 256;

    constexpr int TILE_WIDTH = 8;
    constexpr int TILE_HEIGHT = 8;

    constexpr int TILE_ROWS = 256 / TILE_HEIGHT;
    constexpr int TILE_COLUMNS = 256 / TILE_WIDTH;

    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < TILE_ROWS * TILE_COLUMNS; t++)
    {
        int brow = t / TILE_ROWS;
        int bcol = t % TILE_COLUMNS;

        for (int r = 0; r < TILE_HEIGHT; r++)
        {
            for (int c = 0; c < TILE_WIDTH; c++)
            {
                int row = (TILE_HEIGHT * brow + r);
                int col = (TILE_WIDTH * bcol + c);

                glm::vec2 pixelSampleNDC = generatePixelSampleNDC(col, row, du, dv);

                int irow = (int)(height * (0.5f * (pixelSampleNDC.y + 1.0f)));
                int icol = (int)(width * (0.5f * (pixelSampleNDC.x + 1.0f)));

                irow = glm::min(height - 1, glm::max(0, irow));
                icol = glm::min(width - 1, glm::max(0, icol));

                int offset = width * irow + icol;

                assert(offset >= 0);
                assert(offset < width * height);

                mSamplesPerRay[offset]++;

                // Read color from image
                float red = mImage[3 * offset + 0];
                float green = mImage[3 * offset + 1];
                float blue = mImage[3 * offset + 2];
                glm::vec3 color = glm::vec3(red, green, blue);

                color += computeColorIterative(tlas, blas, materials, getCameraRay(pixelSampleNDC),
                                                maxBounces);

                // Store computed color to image
                mImage[3 * offset + 0] = color.r;
                mImage[3 * offset + 1] = color.g;
                mImage[3 * offset + 2] = color.b;
            }
        }
    }
}

void Camera::updateFinalImage()
{
    // Image size
    int width = getNativeGraphicsRaytracingTex()->getWidth();
    int height = getNativeGraphicsRaytracingTex()->getHeight();

    std::vector<unsigned char> finalImage(3 * width * height);
    #pragma omp parallel for
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            int sampleCount = mSamplesPerRay[width * row + col];

            if (sampleCount > 0)
            {
                // Read color from image
                float r = mImage[3 * width * row + 3 * col + 0];
                float g = mImage[3 * width * row + 3 * col + 1];
                float b = mImage[3 * width * row + 3 * col + 2];

                float scale = 1.0f / sampleCount;
                r *= scale;
                g *= scale;
                b *= scale;

                // Gamma correction
                r = glm::sqrt(r);
                g = glm::sqrt(g);
                b = glm::sqrt(b);

                int ir = (int)(255 * glm::clamp(r, 0.0f, 1.0f));
                int ig = (int)(255 * glm::clamp(g, 0.0f, 1.0f));
                int ib = (int)(255 * glm::clamp(b, 0.0f, 1.0f));

                finalImage[3 * width * row + 3 * col + 0] = static_cast<unsigned char>(ir);
                finalImage[3 * width * row + 3 * col + 1] = static_cast<unsigned char>(ig);
                finalImage[3 * width * row + 3 * col + 2] = static_cast<unsigned char>(ib);
            }
            else
            {
                finalImage[3 * width * row + 3 * col + 0] = static_cast<unsigned char>(0);
                finalImage[3 * width * row + 3 * col + 1] = static_cast<unsigned char>(0);
                finalImage[3 * width * row + 3 * col + 2] = static_cast<unsigned char>(0);
            }
        }
    }

    mTargets.mRaytracingTex->load(finalImage);
}

Ray Camera::getCameraRay(const glm::vec2 &pixelSampleNDC) const
{
    // Transform NDC coordinate back to world space
    glm::vec4 temp = this->getInvViewProjMatrix() * glm::vec4(pixelSampleNDC, 1.0f, 1.0f);
    glm::vec3 pixelCentre_WorldSpace = glm::vec3(temp / temp.w);

    Ray ray;
    ray.mOrigin = this->getPosition();
    ray.mDirection = pixelCentre_WorldSpace - this->getPosition();

    return ray;
}

Ray Camera::getCameraRay(int u, int v, float du, float dv) const
{
    // NDC coordinates for 2x2x2 cube [-1, 1]x[-1, 1]x[-1, 1]
    //         +y |   * +z
    //            |  *
    //            | *
    // -x ________|*________ +x
    // In frustum plane, bottom, left, far corner correspnds to NDC point [-1, -1, 1]
    glm::vec3 farBottomLeft_NDC = glm::vec3(-1.0f, -1.0f, 1.0f);

    // Far, bottom, left corner pixel centre in NDC
    glm::vec3 pixelCentreFarBottomLeft_NDC = farBottomLeft_NDC + glm::vec3(0.5f * du, 0.5f * dv, 0.0f);

    // Far plane pixel centre in NDC
    glm::vec3 pixelCentre_NDC =
        pixelCentreFarBottomLeft_NDC + glm::vec3(u * du, 0.0f, 0.0f) + glm::vec3(0.0f, v * dv, 0.0f);

    // Randomly sample from pixel
    glm::vec3 pixelSample_NDC =
        pixelCentre_NDC + glm::vec3(generate_rand(-0.5f, 0.5f) * du, generate_rand(-0.5f, 0.5f) * dv, 0.0f);

    // Transform NDC coordinate back to world space
    glm::vec4 temp = this->getInvViewProjMatrix() * glm::vec4(pixelSample_NDC, 1.0f);
    glm::vec3 pixelCentre_WorldSpace = glm::vec3(temp / temp.w);

    Ray ray;
    ray.mOrigin = this->getPosition();
    ray.mDirection = pixelCentre_WorldSpace - this->getPosition();

    return ray;
}