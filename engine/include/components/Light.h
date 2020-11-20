#ifndef __LIGHT_H__
#define __LIGHT_H__

#include <GL/glew.h>
#include <gl/gl.h>

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"

#include "Component.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct LightHeader
{
    Guid mComponentId;
    Guid mEntityId;
    glm::vec4 mColor;
    float mIntensity;
    float mSpotAngle;
    float mInnerSpotAngle;
    float mShadowNearPlane;
    float mShadowFarPlane;
    float mShadowAngle;
    float mShadowRadius;
    float mShadowStrength;
    uint8_t mLightType;
    uint8_t mShadowType;
    uint16_t mShadowMapResolution;
};
#pragma pack(pop)

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
    GLuint mShadowCascadeFBO[5];
    GLuint mShadowCascadeDepthTex[5];
    GLuint mShadowSpotlightFBO;
    GLuint mShadowSpotlightDepthTex;
    GLuint mShadowCubemapFBO;
    GLuint mShadowCubemapDepthTex;
};

class Light : public Component
{
  public:
    glm::vec4 mColor;
    float mIntensity;
    float mSpotAngle;
    float mInnerSpotAngle;
    float mShadowNearPlane;
    float mShadowFarPlane;
    float mShadowAngle;
    float mShadowRadius;
    float mShadowStrength;
    LightType mLightType;
    ShadowType mShadowType;

  private:
    bool mIsShadowMapResolutionChanged;
    bool mIsCreated;
    ShadowMapResolution mShadowMapResolution;
    LightTargets mTargets;

  public:
    Light();
    Light(const std::vector<char> &data);
    ~Light();

    std::vector<char> serialize() const;
    std::vector<char> serialize(const Guid &componentId, const Guid &entityId) const;
    void deserialize(const std::vector<char> &data);

    void createTargets();
    void destroyTargets();
    void resizeTargets();

    bool isCreated() const;
    bool isShadowMapResolutionChanged() const;
    void setShadowMapResolution(ShadowMapResolution resolution);
    ShadowMapResolution getShadowMapResolution() const;

    glm::mat4 getProjMatrix() const;

    GLuint getNativeGraphicsShadowCascadeFBO(int index) const;
    GLuint getNativeGraphicsShadowSpotlightFBO() const;
    GLuint getNativeGraphicsShadowCubemapFBO() const;

    GLuint getNativeGraphicsShadowCascadeDepthTex(int index) const;
    GLuint getNativeGrpahicsShadowSpotlightDepthTex() const;
    GLuint getNativeGraphicsShadowCubemapDepthTex() const;
};

template <typename T> struct IsLight
{
    static constexpr bool value = false;
};

template <> struct ComponentType<Light>
{
    static constexpr int type = PhysicsEngine::LIGHT_TYPE;
};
template <> struct IsLight<Light>
{
    static constexpr bool value = true;
};
template <> struct IsComponent<Light>
{
    static constexpr bool value = true;
};
template <> struct IsComponentInternal<Light>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif