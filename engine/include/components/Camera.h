#ifndef CAMERA_H__
#define CAMERA_H__

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
#include "../core/Viewport.h"

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
    Guid mTargetTextureId;

    RenderPath mRenderPath;
    CameraMode mMode;
    CameraSSAO mSSAO;
    CameraGizmos mGizmos;

    Color mBackgroundColor;

    GraphicsQuery mQuery;

  private:
    Frustum mFrustum;
    Viewport mViewport;
    CameraTargets mTargets;

    glm::vec3 mSsaoSamples[64];

    glm::vec3 mPosition;
    glm::mat4 mViewMatrix;
    glm::mat4 mInvViewMatrix;
    glm::mat4 mProjMatrix;

    bool mIsCreated;
    bool mIsViewportChanged;

    std::unordered_map<int, Guid> mColoringMap;

  public:
    Camera();
    Camera(Guid id);
    ~Camera();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void createTargets();
    void destroyTargets();
    void resizeTargets();
    void beginQuery();
    void endQuery();

    void computeViewMatrix(const glm::vec3 &position, const glm::vec3 &forward, const glm::vec3 &up);
    void assignColoring(int color, const Guid &transformId);
    void clearColoring();

    bool isCreated() const;
    bool isViewportChanged() const;
    glm::vec3 getPosition() const;
    glm::mat4 getViewMatrix() const;
    glm::mat4 getInvViewMatrix() const;
    glm::mat4 getProjMatrix() const;
    glm::vec3 getSSAOSample(int sample) const;
    Guid getTransformIdAtScreenPos(int x, int y) const;

    Frustum getFrustum() const;
    Viewport getViewport() const;
    void setFrustum(float fov, float aspectRatio, float nearPlane, float farPlane);
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

template <> struct ComponentType<Camera>
{
    static constexpr int type = CAMERA_TYPE;
};

template <> struct IsComponentInternal<Camera>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

namespace YAML
{
// CameraMode
template <> struct convert<PhysicsEngine::CameraMode>
{
    static Node encode(const PhysicsEngine::CameraMode &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::CameraMode &rhs)
    {
        rhs = static_cast<PhysicsEngine::CameraMode>(node.as<int>());
        return true;
    }
};

// CameraSSAO
template <> struct convert<PhysicsEngine::CameraSSAO>
{
    static Node encode(const PhysicsEngine::CameraSSAO &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::CameraSSAO &rhs)
    {
        rhs = static_cast<PhysicsEngine::CameraSSAO>(node.as<int>());
        return true;
    }
};

// CameraGizmos
template <> struct convert<PhysicsEngine::CameraGizmos>
{
    static Node encode(const PhysicsEngine::CameraGizmos &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::CameraGizmos &rhs)
    {
        rhs = static_cast<PhysicsEngine::CameraGizmos>(node.as<int>());
        return true;
    }
};

// RenderPath
template <> struct convert<PhysicsEngine::RenderPath>
{
    static Node encode(const PhysicsEngine::RenderPath &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::RenderPath &rhs)
    {
        rhs = static_cast<PhysicsEngine::RenderPath>(node.as<int>());
        return true;
    }
};
} // namespace YAML

#endif