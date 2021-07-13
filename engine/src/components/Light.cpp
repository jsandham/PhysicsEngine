#include <algorithm>

#include "../../include/components/Light.h"
#include "../../include/core/Serialization.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Light::Light() : Component()
{
    mEntityId = Guid::INVALID;

    mColor = glm::vec4(0.4f, 0.4f, 0.4f, 1.0f);
    mIntensity = 1.0f;
    mSpotAngle = glm::cos(glm::radians(15.0f));

    mInnerSpotAngle = glm::cos(glm::radians(12.5f));
    mShadowNearPlane = 0.1f;
    mShadowFarPlane = 100.0f;
    mShadowAngle = 0.0f;
    mShadowRadius = 0.0f;
    mShadowStrength = 1.0f;
    mLightType = LightType::Directional;
    mShadowType = ShadowType::Hard;

    mIsCreated = false;
    mIsShadowMapResolutionChanged = false;
    mShadowMapResolution = ShadowMapResolution::Medium1024x1024;

    mTargets.mShadowCascadeFBO[0] = 0;
    mTargets.mShadowCascadeFBO[1] = 0;
    mTargets.mShadowCascadeFBO[2] = 0;
    mTargets.mShadowCascadeFBO[3] = 0;
    mTargets.mShadowCascadeFBO[4] = 0;

    mTargets.mShadowCascadeDepthTex[0] = 0;
    mTargets.mShadowCascadeDepthTex[1] = 0;
    mTargets.mShadowCascadeDepthTex[2] = 0;
    mTargets.mShadowCascadeDepthTex[3] = 0;
    mTargets.mShadowCascadeDepthTex[4] = 0;

    mTargets.mShadowSpotlightFBO = 0;
    mTargets.mShadowSpotlightDepthTex = 0;
    mTargets.mShadowCubemapFBO = 0;
    mTargets.mShadowCubemapDepthTex = 0;
}

Light::Light(Guid id) : Component(id)
{
    mEntityId = Guid::INVALID;

    mColor = glm::vec4(0.4f, 0.4f, 0.4f, 1.0f);
    mIntensity = 1.0f;
    mSpotAngle = glm::cos(glm::radians(15.0f));

    mInnerSpotAngle = glm::cos(glm::radians(12.5f));
    mShadowNearPlane = 0.1f;
    mShadowFarPlane = 100.0f;
    mShadowAngle = 0.0f;
    mShadowRadius = 0.0f;
    mShadowStrength = 1.0f;
    mLightType = LightType::Directional;
    mShadowType = ShadowType::Hard;

    mIsCreated = false;
    mIsShadowMapResolutionChanged = false;
    mShadowMapResolution = ShadowMapResolution::Medium1024x1024;

    mTargets.mShadowCascadeFBO[0] = 0;
    mTargets.mShadowCascadeFBO[1] = 0;
    mTargets.mShadowCascadeFBO[2] = 0;
    mTargets.mShadowCascadeFBO[3] = 0;
    mTargets.mShadowCascadeFBO[4] = 0;

    mTargets.mShadowCascadeDepthTex[0] = 0;
    mTargets.mShadowCascadeDepthTex[1] = 0;
    mTargets.mShadowCascadeDepthTex[2] = 0;
    mTargets.mShadowCascadeDepthTex[3] = 0;
    mTargets.mShadowCascadeDepthTex[4] = 0;

    mTargets.mShadowSpotlightFBO = 0;
    mTargets.mShadowSpotlightDepthTex = 0;
    mTargets.mShadowCubemapFBO = 0;
    mTargets.mShadowCubemapDepthTex = 0;
}

Light::~Light()
{
}

void Light::serialize(YAML::Node &out) const
{
    Component::serialize(out);

    out["color"] = mColor;
    out["intensity"] = mIntensity;
    out["spotAngle"] = mSpotAngle;
    out["innerSpotAngle"] = mInnerSpotAngle;
    out["shadowNearPlane"] = mShadowNearPlane;
    out["shadowFarPlane"] = mShadowFarPlane;
    out["shadowAngle"] = mShadowAngle;
    out["shadowRadius"] = mShadowRadius;
    out["shadowStrength"] = mShadowStrength;
    out["lightType"] = mLightType;
    out["shadowType"] = mShadowType;
    out["shadowMapResolution"] = mShadowMapResolution;
}

void Light::deserialize(const YAML::Node &in)
{
    Component::deserialize(in);

    mColor = YAML::getValue<glm::vec4>(in, "color");
    mIntensity = YAML::getValue<float>(in, "intensity");
    mSpotAngle = YAML::getValue<float>(in, "spotAngle");
    mInnerSpotAngle = YAML::getValue<float>(in, "innerSpotAngle");
    mShadowNearPlane = YAML::getValue<float>(in, "shadowNearPlane");
    mShadowFarPlane = YAML::getValue<float>(in, "shadowFarPlane");
    mShadowAngle = YAML::getValue<float>(in, "shadowAngle");
    mShadowRadius = YAML::getValue<float>(in, "shadowRadius");
    mShadowStrength = YAML::getValue<float>(in, "shadowStrength");
    mLightType = YAML::getValue<LightType>(in, "lightType");
    mShadowType = YAML::getValue<ShadowType>(in, "shadowType");
    mShadowMapResolution = YAML::getValue<ShadowMapResolution>(in, "shadowMapResolution");

    mIsShadowMapResolutionChanged = true;
}

int Light::getType() const
{
    return PhysicsEngine::LIGHT_TYPE;
}

std::string Light::getObjectName() const
{
    return PhysicsEngine::LIGHT_NAME;
}

void Light::createTargets()
{
    Graphics::createTargets(&mTargets, mShadowMapResolution);

    mIsCreated = true;
}

void Light::destroyTargets()
{
    Graphics::destroyTargets(&mTargets);

    mIsCreated = false;
}

void ::Light::resizeTargets()
{
    Graphics::resizeTargets(&mTargets, mShadowMapResolution);

    mIsShadowMapResolutionChanged = false;
}

bool Light::isCreated() const
{
    return mIsCreated;
}

bool Light::isShadowMapResolutionChanged() const
{
    return mIsShadowMapResolutionChanged;
}

void Light::setShadowMapResolution(ShadowMapResolution resolution)
{
    mShadowMapResolution = resolution;

    mIsShadowMapResolutionChanged = true;
}

ShadowMapResolution Light::getShadowMapResolution() const
{
    return mShadowMapResolution;
}

glm::mat4 Light::getProjMatrix() const
{
    return glm::perspective(2.0f * glm::radians(mSpotAngle), 1.0f, 0.1f, 12.0f);
}

GLuint Light::getNativeGraphicsShadowCascadeFBO(int index) const
{
    return mTargets.mShadowCascadeFBO[std::min(4, std::max(0, index))];
}

GLuint Light::getNativeGraphicsShadowSpotlightFBO() const
{
    return mTargets.mShadowSpotlightFBO;
}

GLuint Light::getNativeGraphicsShadowCubemapFBO() const
{
    return mTargets.mShadowCubemapFBO;
}

GLuint Light::getNativeGraphicsShadowCascadeDepthTex(int index) const
{
    return mTargets.mShadowCascadeDepthTex[std::min(4, std::max(0, index))];
}

GLuint Light::getNativeGrpahicsShadowSpotlightDepthTex() const
{
    return mTargets.mShadowSpotlightDepthTex;
}

GLuint Light::getNativeGraphicsShadowCubemapDepthTex() const
{
    return mTargets.mShadowCubemapDepthTex;
}