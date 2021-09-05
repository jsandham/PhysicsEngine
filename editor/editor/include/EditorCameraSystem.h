#ifndef __EDITOR_CAMERA_SYSTEM_H__
#define __EDITOR_CAMERA_SYSTEM_H__

#define GLM_FORCE_RADIANS

#include <vector>

#include <systems/System.h>

#include <components/Camera.h>
#include <components/Transform.h>

#include <core/Input.h>

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct EditorCameraSystemHeader
{
    Guid mSystemId;
    int32_t mUpdateOrder;
};
#pragma pack(pop)

class EditorCameraSystem : public System
{
  private:
    static const float YAW_PAN_SENSITIVITY;
    static const float PITCH_PAN_SENSITIVITY;
    static const float ZOOM_SENSITIVITY;
    static const float TRANSLATE_SENSITIVITY;

  private:
    Transform *mTransform;
    Camera *mCamera;

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

  public:
    EditorCameraSystem(World* world);
    EditorCameraSystem(World* world, Guid id);
    ~EditorCameraSystem();

    virtual void serialize(YAML::Node& out) const override;
    virtual void deserialize(const YAML::Node& in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    /*std::vector<char> serialize() const;
    std::vector<char> serialize(const Guid &systemId) const;
    void deserialize(const std::vector<char> &data);*/

    void init(World *world);
    void update(const Input &input, const Time &time);

    void resetCamera();
    void setViewport(Viewport viewport);
    void setFrustum(Frustum frustum);
    void setRenderPath(RenderPath path);
    void setSSAO(CameraSSAO ssao);
    void setGizmos(CameraGizmos gizmos);

    Viewport getViewport() const;
    Frustum getFrustum() const;
    RenderPath getRenderPath() const;
    CameraSSAO getSSAO() const;
    CameraGizmos getGizmos() const;

    Camera* getCamera() const;

    Guid getTransformUnderMouse(float nx, float ny) const;
    int getMousePosX() const;
    int getMousePosY() const;
    bool isLeftMouseClicked() const;
    bool isRightMouseClicked() const;
    bool isLeftMouseHeldDown() const;
    bool isRightMouseHeldDown() const;
    glm::vec2 distanceTraveledSinceLeftMouseClick() const;
    glm::vec2 distanceTraveledSinceRightMouseClick() const;

    unsigned int getNativeGraphicsMainFBO() const;
    unsigned int getNativeGraphicsColorTex() const;
    unsigned int getNativeGraphicsDepthTex() const;
    unsigned int getNativeGraphicsColorPickingTex() const;
    unsigned int getNativeGraphicsPositionTex() const;
    unsigned int getNativeGraphicsNormalTex() const;
    unsigned int getNativeGraphicsAlbedoSpecTex() const;
    unsigned int getNativeGraphicsSSAOColorTex() const;
    unsigned int getNativeGraphicsSSAONoiseTex() const;

    GraphicsQuery getQuery() const;

    glm::vec3 getCameraForward() const;
    glm::vec3 getCameraPosition() const;
    glm::mat4 getViewMatrix() const;
    glm::mat4 getInvViewMatrix() const;
    glm::mat4 getProjMatrix() const;

    Ray normalizedDeviceSpaceToRay(float x, float y) const;
};

template <> struct SystemType<EditorCameraSystem>
{
    static constexpr int type = 21;
};

} // namespace PhysicsEngine

#endif
