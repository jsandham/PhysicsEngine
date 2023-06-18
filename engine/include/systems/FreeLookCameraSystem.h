#ifndef FREELOOK_CAMERA_SYSTEM_H__
#define FREELOOK_CAMERA_SYSTEM_H__
#include <vector>
#include <string>

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"
#include "glm/gtx/quaternion.hpp"

#include "System.h"
#include "../components/Camera.h"
#include "../components/Transform.h"

namespace PhysicsEngine
{
    struct CameraSystemConfig
    {
        bool mRenderToScreen;
        bool mSpawnCameraOnInit;
    };

    class FreeLookCameraSystem : public System
    {
    private:
        static const float YAW_PAN_SENSITIVITY;
        static const float PITCH_PAN_SENSITIVITY;
        static const float ZOOM_SENSITIVITY;
        static const float TRANSLATE_SENSITIVITY;

    private:
        Guid mTransformId;
        Guid mCameraId;

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
        FreeLookCameraSystem(World *world, const Id &id);
        FreeLookCameraSystem(World *world, const Guid &guid, const Id &id);
        ~FreeLookCameraSystem();

        virtual void serialize(YAML::Node& out) const override;
        virtual void deserialize(const YAML::Node& in) override;

        virtual int getType() const override;
        virtual std::string getObjectName() const override;

        void init(World* world) override;
        void update(const Input& input, const Time& time) override;

        void resetCamera();
        void configureCamera(CameraSystemConfig config);
        void setViewport(const Viewport& viewport);
        void setFrustum(const Frustum& frustum);
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

        Camera* getCamera() const;
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
        RenderTextureHandle *getNativeGraphicsColorTex() const;
        RenderTextureHandle *getNativeGraphicsDepthTex() const;
        RenderTextureHandle *getNativeGraphicsColorPickingTex() const;
        RenderTextureHandle *getNativeGraphicsPositionTex() const;
        RenderTextureHandle *getNativeGraphicsNormalTex() const;
        RenderTextureHandle *getNativeGraphicsAlbedoSpecTex() const;
        RenderTextureHandle *getNativeGraphicsSSAOColorTex() const;
        RenderTextureHandle *getNativeGraphicsSSAONoiseTex() const;

        GraphicsQuery getQuery() const;

        glm::vec3 getCameraForward() const;
        glm::vec3 getCameraPosition() const;
        glm::mat4 getViewMatrix() const;
        glm::mat4 getInvViewMatrix() const;
        glm::mat4 getProjMatrix() const;

        Ray normalizedDeviceSpaceToRay(float x, float y) const;
    };

    template <> struct SystemType<FreeLookCameraSystem>
    {
        static constexpr int type = PhysicsEngine::FREELOOKCAMERASYSTEM_TYPE;
    };
    template <> struct IsSystemInternal<FreeLookCameraSystem>
    {
        static constexpr bool value = true;
    };

} // namespace PhysicsEngine

#endif
