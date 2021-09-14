#ifndef LIGHT_H__
#define LIGHT_H__

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "Component.h"

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
    unsigned int mShadowCascadeFBO[5];
    unsigned int mShadowCascadeDepthTex[5];
    unsigned int mShadowSpotlightFBO;
    unsigned int mShadowSpotlightDepthTex;
    unsigned int mShadowCubemapFBO;
    unsigned int mShadowCubemapDepthTex;
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
    bool mIsShadowMapResolutionChanged;
    bool mIsCreated;
    ShadowMapResolution mShadowMapResolution;
    LightTargets mTargets;

  public:
    Light(World *world);
    Light(World *world, Guid id);
    ~Light();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void createTargets();
    void destroyTargets();
    void resizeTargets();

    bool isCreated() const;
    bool isShadowMapResolutionChanged() const;
    void setShadowMapResolution(ShadowMapResolution resolution);
    ShadowMapResolution getShadowMapResolution() const;

    glm::mat4 getProjMatrix() const;

    unsigned int getNativeGraphicsShadowCascadeFBO(int index) const;
    unsigned int getNativeGraphicsShadowSpotlightFBO() const;
    unsigned int getNativeGraphicsShadowCubemapFBO() const;

    unsigned int getNativeGraphicsShadowCascadeDepthTex(int index) const;
    unsigned int getNativeGrpahicsShadowSpotlightDepthTex() const;
    unsigned int getNativeGraphicsShadowCubemapDepthTex() const;
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