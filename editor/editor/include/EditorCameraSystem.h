#ifndef __EDITOR_CAMERA_SYSTEM_H__
#define __EDITOR_CAMERA_SYSTEM_H__

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
    EditorCameraSystem();
    EditorCameraSystem(std::vector<char> data);
    ~EditorCameraSystem();

    std::vector<char> serialize() const;
    std::vector<char> serialize(Guid systemId) const;
    void deserialize(const std::vector<char> &data);

    void init(World *world);
    void update(Input input, Time time);

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

    Guid getTransformUnderMouse(float nx, float ny) const;
    int getMousePosX() const;
    int getMousePosY() const;
    bool isLeftMouseClicked() const;
    bool isRightMouseClicked() const;
    bool isLeftMouseHeldDown() const;
    bool isRightMouseHeldDown() const;
    glm::vec2 distanceTraveledSinceLeftMouseClick() const;
    glm::vec2 distanceTraveledSinceRightMouseClick() const;

    GLuint getNativeGraphicsMainFBO() const;
    GLuint getNativeGraphicsColorTex() const;
    GLuint getNativeGraphicsDepthTex() const;
    GLuint getNativeGraphicsColorPickingTex() const;
    GLuint getNativeGraphicsPositionTex() const;
    GLuint getNativeGraphicsNormalTex() const;
    GLuint getNativeGraphicsAlbedoSpecTex() const;
    GLuint getNativeGraphicsSSAOColorTex() const;
    GLuint getNativeGraphicsSSAONoiseTex() const;

    GraphicsQuery getQuery() const;

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjMatrix() const;

    Ray normalizedDeviceSpaceToRay(float x, float y) const;
};

template <typename T> struct IsEditorCameraSystem
{
    static constexpr bool value = false;
};

template <> struct SystemType<EditorCameraSystem>
{
    static constexpr int type = 21;
};
template <> struct IsEditorCameraSystem<EditorCameraSystem>
{
    static constexpr bool value = true;
};
template <> struct IsSystem<EditorCameraSystem>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif
