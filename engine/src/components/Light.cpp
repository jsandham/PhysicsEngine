#include <algorithm>

#include "../../include/components/Light.h"
#include "../../include/graphics/Renderer.h"

using namespace PhysicsEngine;

Light::Light(World *world, const Id &id) : Component(world, id)
{
    mColor = glm::vec4(0.4f, 0.4f, 0.4f, 1.0f);
    mIntensity = 1.0f;
    mSpotAngle = 15.0f;// glm::cos(glm::radians(15.0f));

    mInnerSpotAngle = 12.5f;// glm::cos(glm::radians(12.5f));
    mShadowStrength = 1.0f;
    mShadowNearPlane = 1.0f;
    mShadowFarPlane = 100.0f;
    mShadowBias = 0.005f;
    mLightType = LightType::Directional;
    mShadowType = ShadowType::Hard;

    mEnabled = true;
    mShadowMapResolution = ShadowMapResolution::Medium1024x1024;

    mTargets.mShadowCascadeFBO[0] = Framebuffer::create(1024, 1024);
    mTargets.mShadowCascadeFBO[1] = Framebuffer::create(1024, 1024);
    mTargets.mShadowCascadeFBO[2] = Framebuffer::create(1024, 1024);
    mTargets.mShadowCascadeFBO[3] = Framebuffer::create(1024, 1024);
    mTargets.mShadowCascadeFBO[4] = Framebuffer::create(1024, 1024);

    mTargets.mShadowSpotlightFBO = Framebuffer::create(1024, 1024);
    mTargets.mShadowCubemapFBO = Framebuffer::create(1024, 1024);
}

Light::Light(World *world, const Guid &guid, const Id &id) : Component(world, guid, id)
{
    mColor = glm::vec4(0.4f, 0.4f, 0.4f, 1.0f);
    mIntensity = 1.0f;
    mSpotAngle = 15.0f;// glm::cos(glm::radians(15.0f));

    mInnerSpotAngle = 12.5f;//glm::cos(glm::radians(12.5f));
    mShadowStrength = 1.0f;
    mShadowNearPlane = 1.0f;
    mShadowFarPlane = 100.0f;
    mShadowBias = 0.005f;
    mLightType = LightType::Directional;
    mShadowType = ShadowType::Hard;

    mEnabled = true;
    mShadowMapResolution = ShadowMapResolution::Medium1024x1024;

    mTargets.mShadowCascadeFBO[0] = Framebuffer::create(1024, 1024);
    mTargets.mShadowCascadeFBO[1] = Framebuffer::create(1024, 1024);
    mTargets.mShadowCascadeFBO[2] = Framebuffer::create(1024, 1024);
    mTargets.mShadowCascadeFBO[3] = Framebuffer::create(1024, 1024);
    mTargets.mShadowCascadeFBO[4] = Framebuffer::create(1024, 1024);

    mTargets.mShadowSpotlightFBO = Framebuffer::create(1024, 1024);
    mTargets.mShadowCubemapFBO = Framebuffer::create(1024, 1024);
}

Light::~Light()
{
    delete mTargets.mShadowCascadeFBO[0];
    delete mTargets.mShadowCascadeFBO[1];
    delete mTargets.mShadowCascadeFBO[2];
    delete mTargets.mShadowCascadeFBO[3];
    delete mTargets.mShadowCascadeFBO[4];

    delete mTargets.mShadowSpotlightFBO;
    delete mTargets.mShadowCubemapFBO;
}

void Light::serialize(YAML::Node &out) const
{
    Component::serialize(out);

    out["color"] = mColor;
    out["intensity"] = mIntensity;
    out["spotAngle"] = mSpotAngle;
    out["innerSpotAngle"] = mInnerSpotAngle;
    out["shadowStrength"] = mShadowStrength;
    out["shadowNearPlane"] = mShadowNearPlane;
    out["shadowFarPlane"] = mShadowFarPlane;
    out["shadowBias"] = mShadowBias;
    out["lightType"] = mLightType;
    out["shadowType"] = mShadowType;
    out["shadowMapResolution"] = mShadowMapResolution;
    out["enabled"] = mEnabled;
}

void Light::deserialize(const YAML::Node &in)
{
    Component::deserialize(in);

    mColor = YAML::getValue<glm::vec4>(in, "color");
    mIntensity = YAML::getValue<float>(in, "intensity");
    mSpotAngle = YAML::getValue<float>(in, "spotAngle");
    mInnerSpotAngle = YAML::getValue<float>(in, "innerSpotAngle");
    mShadowStrength = YAML::getValue<float>(in, "shadowStrength");
    mShadowNearPlane = YAML::getValue<float>(in, "shadowNearPlane");
    mShadowFarPlane = YAML::getValue<float>(in, "shadowFarPlane");
    mShadowBias = YAML::getValue<float>(in, "shadowBias");
    mLightType = YAML::getValue<LightType>(in, "lightType");
    mShadowType = YAML::getValue<ShadowType>(in, "shadowType");
    mShadowMapResolution = YAML::getValue<ShadowMapResolution>(in, "shadowMapResolution");
    mEnabled = YAML::getValue<bool>(in, "enabled");
}

int Light::getType() const
{
    return PhysicsEngine::LIGHT_TYPE;
}

std::string Light::getObjectName() const
{
    return PhysicsEngine::LIGHT_NAME;
}

void ::Light::resizeTargets()
{
    //Renderer::getRenderer()->resizeTargets(&mTargets, mShadowMapResolution);
}

void Light::setShadowMapResolution(ShadowMapResolution resolution)
{
    mShadowMapResolution = resolution;

    resizeTargets();
}

ShadowMapResolution Light::getShadowMapResolution() const
{
    return mShadowMapResolution;
}

glm::mat4 Light::getProjMatrix() const
{
    if (mLightType == LightType::Spot)
    {
        return glm::perspective(2.0f * glm::radians(mSpotAngle), 1.0f, mShadowNearPlane, mShadowFarPlane);
    }
    else if (mLightType == LightType::Point) 
    {
        return glm::perspective(glm::radians(90.0f), 1.0f, mShadowNearPlane, mShadowFarPlane);
    }

    return glm::mat4(1.0f);
}

Framebuffer* Light::getNativeGraphicsShadowCascadeFBO(int index) const
{
    return mTargets.mShadowCascadeFBO[std::min(4, std::max(0, index))];
}

Framebuffer* Light::getNativeGraphicsShadowSpotlightFBO() const
{
    return mTargets.mShadowSpotlightFBO;
}

Framebuffer* Light::getNativeGraphicsShadowCubemapFBO() const
{
    return mTargets.mShadowCubemapFBO;
}

TextureHandle *Light::getNativeGraphicsShadowCascadeDepthTex(int index) const
{
    return mTargets.mShadowCascadeFBO[std::min(4, std::max(0, index))]->getDepthTex();
}

TextureHandle *Light::getNativeGrpahicsShadowSpotlightDepthTex() const
{
    return mTargets.mShadowSpotlightFBO->getDepthTex();
}

TextureHandle *Light::getNativeGraphicsShadowCubemapDepthTex() const
{
    return mTargets.mShadowCubemapFBO->getDepthTex();
}