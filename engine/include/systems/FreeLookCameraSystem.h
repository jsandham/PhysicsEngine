#ifndef FREELOOK_CAMERA_SYSTEM_H__
#define FREELOOK_CAMERA_SYSTEM_H__
#include <string>
#include <vector>

#include "../core/glm.h"
#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"

#include "../components/Camera.h"
#include "../components/Transform.h"

namespace PhysicsEngine
{
struct CameraSystemConfig
{
    bool mRenderToScreen;
    bool mSpawnCameraOnInit;
};

class World;

class FreeLookCameraSystem
{
  private:
    static const float YAW_PAN_SENSITIVITY;
    static const float PITCH_PAN_SENSITIVITY;
    static const float ZOOM_SENSITIVITY;
    static const float TRANSLATE_SENSITIVITY;

  private:
    Guid mGuid;
    Id mId;
    World* mWorld;

    Guid mTransformGuid;
    Guid mCameraGuid;

    int mMousePosX;
    int mMousePosY;
    int mMousePosXOnLeftClick;
    int mMousePosYOnLeftClick;
    int mMousePosXOnRightClick;
    int mMousePosYOnRightClick;
    bool mIsLeftMouseClicked;
    bool mIsRightMouseClicked;
    bool mIsLeftMouseHeldDown;
    bool mIsRightMouseHeldDown;
    glm::quat rotationOnClick;

    bool mRenderToScreen;
    bool mSpawnCameraOnInit;

  public:
    HideFlag mHide;
    bool mEnabled;

  public:
    FreeLookCameraSystem(World *world, const Id &id);
    FreeLookCameraSystem(World *world, const Guid &guid, const Id &id);
    ~FreeLookCameraSystem();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getGuid() const;
    Id getId() const;

    void init(World *world);
    void update();

    void resetCamera();
    void configureCamera(CameraSystemConfig config);
    void setViewport(const Viewport &viewport);
    void setFrustum(const Frustum &frustum);
    void setViewport(int x, int y, int width, int height);
    void setFrustum(float fov, float aspectRatio, float near, float far);
    void setRenderPath(RenderPath path);
    void setSSAO(CameraSSAO ssao);
    void setGizmos(CameraGizmos gizmos);

    Viewport getViewport() const;
    Frustum getFrustum() const;
    RenderPath getRenderPath() const;
    CameraSSAO getSSAO() const;
    CameraGizmos getGizmos() const;

    Camera *getCamera() const;
    Transform *getTransform() const;

    Id getTransformUnderMouse(float nx, float ny) const;
    int getMousePosX() const;
    int getMousePosY() const;
    bool isLeftMouseClicked() const;
    bool isRightMouseClicked() const;
    bool isLeftMouseHeldDown() const;
    bool isRightMouseHeldDown() const;
    glm::vec2 distanceTraveledSinceLeftMouseClick() const;
    glm::vec2 distanceTraveledSinceRightMouseClick() const;

    Framebuffer *getNativeGraphicsMainFBO() const;
    RenderTextureHandle *getNativeGraphicsRaytracingTex() const;
    RenderTextureHandle *getNativeGraphicsRaytracingIntersectionCountTex() const;
    RenderTextureHandle *getNativeGraphicsColorTex() const;
    RenderTextureHandle *getNativeGraphicsDepthTex() const;
    RenderTextureHandle *getNativeGraphicsColorPickingTex() const;
    RenderTextureHandle *getNativeGraphicsPositionTex() const;
    RenderTextureHandle *getNativeGraphicsNormalTex() const;
    RenderTextureHandle *getNativeGraphicsAlbedoSpecTex() const;
    RenderTextureHandle *getNativeGraphicsSSAOColorTex() const;
    RenderTextureHandle *getNativeGraphicsSSAONoiseTex() const;
    RenderTextureHandle *getNativeGraphicsOcclusionMapTex() const;

    TimingQuery getQuery() const;

    glm::vec3 getCameraForward() const;
    glm::vec3 getCameraPosition() const;
    glm::mat4 getViewMatrix() const;
    glm::mat4 getInvViewMatrix() const;
    glm::mat4 getProjMatrix() const;

    Ray normalizedDeviceSpaceToRay(float x, float y) const;
};

} // namespace PhysicsEngine

#endif
