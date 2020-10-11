#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <unordered_map>
#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#undef NEAR
#undef FAR
#undef near
#undef far

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"
#include "../glm/gtc/type_ptr.hpp"

#include "Component.h"

#include "../core/Color.h"
#include "../core/Frustum.h"
#include "../core/Ray.h"

#include "../graphics/GraphicsQuery.h"

namespace PhysicsEngine
{
enum class CameraMode
{
    Main,
    Secondary
};

enum class CameraSSAO
{
    SSAO_On,
    SSAO_Off,
};

enum class CameraGizmos
{
    Gizmos_On,
    Gizmos_Off,
};

enum class RenderPath
{
    Forward,
    Deferred
};

#pragma pack(push, 1)
struct CameraHeader
{
    Guid mComponentId;
    Guid mEntityId;
    Guid mTargetTextureId;
    glm::vec4 mBackgroundColor;
    int32_t mX;
    int32_t mY;
    int32_t mWidth;
    int32_t mHeight;
    float mFov;
    float mAspectRatio;
    float mNearPlane;
    float mFarPlane;
    uint8_t mRenderPath;
    uint8_t mMode;
    uint8_t mSSAO;
    uint8_t mGizmos;
};
#pragma pack(pop)

struct Viewport
{
    int mX;
    int mY;
    int mWidth;
    int mHeight;
};

struct CameraTargets
{
    GLuint mMainFBO;
    GLuint mColorTex;
    GLuint mDepthTex;

    GLuint mColorPickingFBO;
    GLuint mColorPickingTex;
    GLuint mColorPickingDepthTex;

    GLuint mGeometryFBO;
    GLuint mPositionTex;
    GLuint mNormalTex;
    GLuint mAlbedoSpecTex;

    GLuint mSsaoFBO;
    GLuint mSsaoColorTex;
    GLuint mSsaoNoiseTex;
};

class Camera : public Component
{
  public:
    Frustum mFrustum;
    Guid mTargetTextureId;

    RenderPath mRenderPath;
    CameraMode mMode;
    CameraSSAO mSSAO;
    CameraGizmos mGizmos;

    Color mBackgroundColor;

    GraphicsQuery mQuery;

  private:
    Viewport mViewport;
    CameraTargets mTargets;

    glm::vec3 mSsaoSamples[64];
    glm::mat4 viewMatrix;

    bool mIsCreated;
    bool mIsViewportChanged;

    std::unordered_map<int, Guid> mColoringMap;

  public:
    Camera();
    Camera(const std::vector<char> &data);
    ~Camera();

    std::vector<char> serialize() const;
    std::vector<char> serialize(Guid componentId, Guid entityId) const;
    void deserialize(const std::vector<char> &data);

    void createTargets();
    void destroyTargets();
    void resizeTargets();
    void beginQuery();
    void endQuery();

    void computeViewMatrix(glm::vec3 position, glm::vec3 forward, glm::vec3 up);
    void assignColoring(int color, Guid transformId);
    void clearColoring();

    bool isCreated() const;
    bool isViewportChanged() const;
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjMatrix() const;
    glm::vec3 getSSAOSample(int sample) const;
    Guid getTransformIdAtScreenPos(int x, int y) const;

    Viewport getViewport() const;
    void setViewport(int x, int y, int width, int height);

    Ray normalizedDeviceSpaceToRay(float x, float y) const;
    Ray screenSpaceToRay(int x, int y) const;

    GLuint getNativeGraphicsMainFBO() const;
    GLuint getNativeGraphicsColorPickingFBO() const;
    GLuint getNativeGraphicsGeometryFBO() const;
    GLuint getNativeGraphicsSSAOFBO() const;

    GLuint getNativeGraphicsColorTex() const;
    GLuint getNativeGraphicsDepthTex() const;
    GLuint getNativeGraphicsColorPickingTex() const;
    GLuint getNativeGraphicsPositionTex() const;
    GLuint getNativeGraphicsNormalTex() const;
    GLuint getNativeGraphicsAlbedoSpecTex() const;
    GLuint getNativeGraphicsSSAOColorTex() const;
    GLuint getNativeGraphicsSSAONoiseTex() const;
};

template <typename T> struct IsCamera
{
    static constexpr bool value = false;
};

template <> struct ComponentType<Camera>
{
    static constexpr int type = CAMERA_TYPE;
};
template <> struct IsCamera<Camera>
{
    static constexpr bool value = true;
};
template <> struct IsComponent<Camera>
{
    static constexpr bool value = true;
};
template <> struct IsComponentInternal<Camera>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif