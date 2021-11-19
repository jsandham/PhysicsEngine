#ifndef CAMERA_H__
#define CAMERA_H__

#include <unordered_map>
#include <vector>

#undef NEAR
#undef FAR
#undef near
#undef far

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "Component.h"

#include "../core/Color.h"
#include "../core/Frustum.h"
#include "../core/Ray.h"
#include "../core/Viewport.h"
#include "../core/RenderTexture.h"

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

enum class ColorTarget
{
    Color,
    Normal,
    Position,
    LinearDepth,
    ShadowCascades
};

enum class ShadowCascades
{
    NoCascades = 0,
    TwoCascades = 1,
    ThreeCascades = 2,
    FourCascades = 3,
    FiveCascades = 4,
};

struct CameraTargets
{
    unsigned int mMainFBO;
    unsigned int mColorTex;
    unsigned int mDepthTex;

    unsigned int mColorPickingFBO;
    unsigned int mColorPickingTex;
    unsigned int mColorPickingDepthTex;

    unsigned int mGeometryFBO;
    unsigned int mPositionTex;
    unsigned int mNormalTex;
    unsigned int mAlbedoSpecTex;

    unsigned int mSsaoFBO;
    unsigned int mSsaoColorTex;
    unsigned int mSsaoNoiseTex;
};

class Camera : public Component
{
  public:
    Guid mRenderTextureId;

    RenderPath mRenderPath;
    ColorTarget mColorTarget;
    CameraMode mMode;
    CameraSSAO mSSAO;
    CameraGizmos mGizmos;
    ShadowCascades mShadowCascades;

    Color mBackgroundColor;

    GraphicsQuery mQuery;

    bool mEnabled;
    bool mRenderToScreen;

  private:
    Frustum mFrustum;
    Viewport mViewport;
    CameraTargets mTargets;

    glm::vec3 mSsaoSamples[64];
    std::array<int, 5> mCascadeSplits;

    glm::mat4 mViewMatrix;
    glm::mat4 mInvViewMatrix;
    glm::mat4 mProjMatrix;
    glm::vec3 mPosition;

    std::unordered_map<Color32, Guid> mColoringMap;

    bool mIsCreated;
    bool mIsViewportChanged;

  public:
    Camera(World *world);
    Camera(World *world, Guid id);
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
    void assignColoring(Color32 color, const Guid& transformId);
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

    std::array<int, 5> getCascadeSplits() const;
    void setCascadeSplit(size_t splitIndex, int splitValue);
    std::array<float, 6> calcViewSpaceCascadeEnds() const;
    std::array<Frustum, 5> calcCascadeFrustums(const std::array<float, 6>& cascadeEnds) const;

    Ray normalizedDeviceSpaceToRay(float x, float y) const;
    Ray screenSpaceToRay(int x, int y) const;

    unsigned int getNativeGraphicsMainFBO() const;
    unsigned int getNativeGraphicsColorPickingFBO() const;
    unsigned int getNativeGraphicsGeometryFBO() const;
    unsigned int getNativeGraphicsSSAOFBO() const;

    unsigned int getNativeGraphicsColorTex() const;
    unsigned int getNativeGraphicsDepthTex() const;
    unsigned int getNativeGraphicsColorPickingTex() const;
    unsigned int getNativeGraphicsPositionTex() const;
    unsigned int getNativeGraphicsNormalTex() const;
    unsigned int getNativeGraphicsAlbedoSpecTex() const;
    unsigned int getNativeGraphicsSSAOColorTex() const;
    unsigned int getNativeGraphicsSSAONoiseTex() const;
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

// ColorTarget
template <> struct convert<PhysicsEngine::ColorTarget>
{
    static Node encode(const PhysicsEngine::ColorTarget& rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::ColorTarget &rhs)
    {
        rhs = static_cast<PhysicsEngine::ColorTarget>(node.as<int>());
        return true;
    }
};

// ShadowCascades
template <> struct convert<PhysicsEngine::ShadowCascades>
{
    static Node encode(const PhysicsEngine::ShadowCascades& rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node& node, PhysicsEngine::ShadowCascades& rhs)
    {
        rhs = static_cast<PhysicsEngine::ShadowCascades>(node.as<int>());
        return true;
    }
};
} // namespace YAML

#endif