#ifndef LIGHT_H__
#define LIGHT_H__

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "Component.h"

#include "../graphics/Framebuffer.h"
#include "../graphics/TextureHandle.h"

namespace PhysicsEngine
{
enum class LightType
{
    Directional,
    Spot,
    Point,
    None
};

enum class ShadowType
{
    Hard,
    Soft,
    None
};

enum class ShadowMapResolution
{
    Low512x512 = 512,
    Medium1024x1024 = 1024,
    High2048x2048 = 2048,
    VeryHigh4096x4096 = 4096
};

struct LightTargets
{
    Framebuffer *mShadowCascadeFBO[5];
    Framebuffer *mShadowSpotlightFBO;
    Framebuffer *mShadowCubemapFBO;
};

class Light : public Component
{
  public:
    glm::vec4 mColor;
    float mIntensity;
    float mSpotAngle;
    float mInnerSpotAngle;
    float mShadowStrength;
    float mShadowNearPlane;
    float mShadowFarPlane;
    float mShadowBias;
    LightType mLightType;
    ShadowType mShadowType;
    bool mEnabled;

  private:
    ShadowMapResolution mShadowMapResolution;
    LightTargets mTargets;

  public:
    Light(World *world, const Id &id);
    Light(World *world, const Guid &guid, const Id &id);
    ~Light();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void resizeTargets();

    bool isShadowMapResolutionChanged() const;
    void setShadowMapResolution(ShadowMapResolution resolution);
    ShadowMapResolution getShadowMapResolution() const;

    glm::mat4 getProjMatrix() const;

    Framebuffer* getNativeGraphicsShadowCascadeFBO(int index) const;
    Framebuffer* getNativeGraphicsShadowSpotlightFBO() const;
    Framebuffer* getNativeGraphicsShadowCubemapFBO() const;

    TextureHandle* getNativeGraphicsShadowCascadeDepthTex(int index) const;
    TextureHandle* getNativeGrpahicsShadowSpotlightDepthTex() const;
    TextureHandle* getNativeGraphicsShadowCubemapDepthTex() const;
};

template <> struct ComponentType<Light>
{
    static constexpr int type = PhysicsEngine::LIGHT_TYPE;
};

template <> struct IsComponentInternal<Light>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

namespace YAML
{
// LightType
template <> struct convert<PhysicsEngine::LightType>
{
    static Node encode(const PhysicsEngine::LightType &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::LightType &rhs)
    {
        rhs = static_cast<PhysicsEngine::LightType>(node.as<int>());
        return true;
    }
};

// ShadowType
template <> struct convert<PhysicsEngine::ShadowType>
{
    static Node encode(const PhysicsEngine::ShadowType &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::ShadowType &rhs)
    {
        rhs = static_cast<PhysicsEngine::ShadowType>(node.as<int>());
        return true;
    }
};

// ShadowMapResolution
template <> struct convert<PhysicsEngine::ShadowMapResolution>
{
    static Node encode(const PhysicsEngine::ShadowMapResolution &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::ShadowMapResolution &rhs)
    {
        rhs = static_cast<PhysicsEngine::ShadowMapResolution>(node.as<int>());
        return true;
    }
};
} // namespace YAML

#endif