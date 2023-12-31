#ifndef CAMERA_H__
#define CAMERA_H__

#include <unordered_map>

#undef NEAR
#undef FAR
#undef near
#undef far

#include "../core/glm.h"
#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/Color.h"
#include "../core/Frustum.h"
#include "../core/Ray.h"
#include "../core/RenderTexture.h"
#include "../core/Viewport.h"

#include "../graphics/Framebuffer.h"
#include "../graphics/TextureHandle.h"
#include "../graphics/GraphicsQuery.h"
#include "../graphics/RenderTextureHandle.h"

#include "ComponentEnums.h"

namespace PhysicsEngine
{
struct CameraTargets
{
    Framebuffer *mMainFBO;
    Framebuffer *mColorPickingFBO;
    Framebuffer *mGeometryFBO;
    Framebuffer *mSsaoFBO;
    Framebuffer *mOcclusionMapFBO;

    RenderTextureHandle *mRaytracingTex;
};

class World;
class Entity;

class Camera
{
  public:
    HideFlag mHide;

    Guid mRenderTextureId;

    RenderPath mRenderPath;
    ColorTarget mColorTarget;
    CameraMode mMode;
    CameraSSAO mSSAO;
    CameraGizmos mGizmos;
    ShadowCascades mShadowCascades;

    Color mBackgroundColor;

    TimingQuery mQuery;

    bool mEnabled;
    bool mRenderToScreen;

  private:
    Guid mGuid;
    Id mId;
    Guid mEntityGuid;

    World *mWorld;

    Frustum mFrustum;
    Viewport mViewport;

    CameraTargets mTargets;

    glm::vec3 mSsaoSamples[64];
    std::array<int, 5> mCascadeSplits;

    glm::mat4 mViewMatrix;
    glm::mat4 mInvViewMatrix;
    glm::mat4 mProjMatrix;
    glm::mat4 mInvProjMatrix;
    glm::mat4 mViewProjMatrix;
    glm::mat4 mInvViewProjMatrix;

    glm::vec3 mPosition;
    glm::vec3 mForward;
    glm::vec3 mUp;
    glm::vec3 mRight;

    std::vector<Id> mColoringIds;

    bool mIsViewportChanged;
    bool mMoved;

  public:
    Camera(World *world, const Id &id);
    Camera(World *world, const Guid &guid, const Id &id);
    ~Camera();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getEntityGuid() const;
    Guid getGuid() const;
    Id getId() const;

    void resizeTargets();
    void beginQuery();
    void endQuery();

    void computeViewMatrix(const glm::vec3 &position, const glm::vec3 &forward, const glm::vec3 &up,
                           const glm::vec3 &right);
    void setColoringIds(const std::vector<Id> &ids);

    bool isCreated() const;
    bool isViewportChanged() const;
    bool moved() const;
    glm::vec3 getPosition() const;
    glm::vec3 getForward() const;
    glm::vec3 getUp() const;
    glm::vec3 getRight() const;
    glm::mat4 getViewMatrix() const;
    glm::mat4 getInvViewMatrix() const;
    glm::mat4 getProjMatrix() const;
    glm::mat4 getInvProjMatrix() const;
    glm::mat4 getViewProjMatrix() const;
    glm::mat4 getInvViewProjMatrix() const;

    glm::vec3 getSSAOSample(int sample) const;
    Id getTransformIdAtScreenPos(int x, int y) const;

    Frustum getFrustum() const;
    Viewport getViewport() const;
    void setFrustum(float fov, float aspectRatio, float nearPlane, float farPlane);
    void setViewport(int x, int y, int width, int height);

    std::array<int, 5> getCascadeSplits() const;
    void setCascadeSplit(size_t splitIndex, int splitValue);
    std::array<float, 6> calcViewSpaceCascadeEnds() const;
    std::array<Frustum, 5> calcCascadeFrustums(const std::array<float, 6> &cascadeEnds) const;

    Ray normalizedDeviceSpaceToRay(float x, float y) const;
    Ray screenSpaceToRay(int x, int y) const;

    Framebuffer *getNativeGraphicsMainFBO() const;
    Framebuffer *getNativeGraphicsColorPickingFBO() const;
    Framebuffer *getNativeGraphicsGeometryFBO() const;
    Framebuffer *getNativeGraphicsSSAOFBO() const;
    Framebuffer *getNativeGraphicsOcclusionMapFBO() const;
    RenderTextureHandle *getNativeGraphicsRaytracingTex() const;

    RenderTextureHandle *getNativeGraphicsColorTex() const;
    RenderTextureHandle *getNativeGraphicsDepthTex() const;
    RenderTextureHandle *getNativeGraphicsColorPickingTex() const;
    RenderTextureHandle *getNativeGraphicsPositionTex() const;
    RenderTextureHandle *getNativeGraphicsNormalTex() const;
    RenderTextureHandle *getNativeGraphicsAlbedoSpecTex() const;
    RenderTextureHandle *getNativeGraphicsSSAOColorTex() const;
    RenderTextureHandle *getNativeGraphicsSSAONoiseTex() const;
    RenderTextureHandle *getNativeGraphicsOcclusionMapTex() const;

    Entity *getEntity() const;

    template <typename T> T *getComponent() const
    {
        return mWorld->getActiveScene()->getComponent<T>(mEntityGuid);
    }

    Ray getCameraRay(int u, int v, float du, float dv) const;
    void updateRayTracingTexture(const std::vector<unsigned char> &data);

  private:
    friend class Scene;
};

} // namespace PhysicsEngine

#endif