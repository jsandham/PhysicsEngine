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

void Light::serialize(std::ostream &out) const
{
    Component::serialize(out);

    PhysicsEngine::write<glm::vec4>(out, mColor);
    PhysicsEngine::write<float>(out, mIntensity);
    PhysicsEngine::write<float>(out, mSpotAngle);
    PhysicsEngine::write<float>(out, mInnerSpotAngle);
    PhysicsEngine::write<float>(out, mShadowNearPlane);
    PhysicsEngine::write<float>(out, mShadowFarPlane);
    PhysicsEngine::write<float>(out, mShadowAngle);
    PhysicsEngine::write<float>(out, mShadowRadius);
    PhysicsEngine::write<float>(out, mShadowStrength);
    PhysicsEngine::write<LightType>(out, mLightType);
    PhysicsEngine::write<ShadowType>(out, mShadowType);
    PhysicsEngine::write<ShadowMapResolution>(out, mShadowMapResolution);
}

void Light::deserialize(std::istream &in)
{
    Component::deserialize(in);

    PhysicsEngine::read<glm::vec4>(in, mColor);
    PhysicsEngine::read<float>(in, mIntensity);
    PhysicsEngine::read<float>(in, mSpotAngle);
    PhysicsEngine::read<float>(in, mInnerSpotAngle);
    PhysicsEngine::read<float>(in, mShadowNearPlane);
    PhysicsEngine::read<float>(in, mShadowFarPlane);
    PhysicsEngine::read<float>(in, mShadowAngle);
    PhysicsEngine::read<float>(in, mShadowRadius);
    PhysicsEngine::read<float>(in, mShadowStrength);
    PhysicsEngine::read<LightType>(in, mLightType);
    PhysicsEngine::read<ShadowType>(in, mShadowType);
    PhysicsEngine::read<ShadowMapResolution>(in, mShadowMapResolution);

    mIsShadowMapResolutionChanged = true;
}

void Light::serialize(YAML::Node& out) const
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

void Light::deserialize(const YAML::Node& in)
{
    Component::deserialize(in);

    mColor = in["color"].as<glm::vec4>();
    mIntensity = in["intensity"].as<float>();
    mSpotAngle = in["spotAngle"].as<float>();
    mInnerSpotAngle = in["innerSpotAngle"].as<float>();
    mShadowNearPlane = in["shadowNearPlane"].as<float>();
    mShadowFarPlane = in["shadowFarPlane"].as<float>();
    mShadowAngle = in["shadowAngle"].as<float>();
    mShadowRadius = in["shadowRadius"].as<float>();
    mShadowStrength = in["shadowStrength"].as<float>();
    mLightType = in["lightType"].as<LightType>();
    mShadowType = in["shadowType"].as<ShadowType>();
    mShadowMapResolution = in["shadowMapResolution"].as<ShadowMapResolution>();

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