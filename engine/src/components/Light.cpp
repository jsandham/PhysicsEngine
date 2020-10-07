#include <algorithm>

#include "../../include/components/Light.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Light::Light()
{
    mComponentId = Guid::INVALID;
    mEntityId = Guid::INVALID;

    mColor = glm::vec4(0.4f, 0.4f, 0.4f, 1.0f);
    mIntensity = 1.0f;
    mSpotAngle = glm::cos(glm::radians(15.0f));
    ;
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

Light::Light(const std::vector<char> &data)
{
    deserialize(data);

    mIsCreated = false;
}

Light::~Light()
{
}

std::vector<char> Light::serialize() const
{
    return serialize(mComponentId, mEntityId);
}

std::vector<char> Light::serialize(Guid componentId, Guid entityId) const
{
    LightHeader header;
    header.mComponentId = componentId;
    header.mEntityId = entityId;
    header.mColor = mColor;
    header.mIntensity = mIntensity;
    header.mSpotAngle = mSpotAngle;
    header.mInnerSpotAngle = mInnerSpotAngle;
    header.mShadowNearPlane = mShadowNearPlane;
    header.mShadowFarPlane = mShadowFarPlane;
    header.mShadowAngle = mShadowAngle;
    header.mShadowRadius = mShadowRadius;
    header.mShadowStrength = mShadowStrength;
    header.mLightType = static_cast<uint8_t>(mLightType);
    header.mShadowType = static_cast<uint8_t>(mShadowType);
    header.mShadowMapResolution = static_cast<uint16_t>(mShadowMapResolution);

    std::vector<char> data(sizeof(LightHeader));

    memcpy(&data[0], &header, sizeof(LightHeader));

    return data;
}

void Light::deserialize(const std::vector<char> &data)
{
    const LightHeader *header = reinterpret_cast<const LightHeader *>(&data[0]);

    mComponentId = header->mComponentId;
    mEntityId = header->mEntityId;
    mColor = header->mColor;
    mIntensity = header->mIntensity;
    mSpotAngle = header->mSpotAngle;
    mInnerSpotAngle = header->mInnerSpotAngle;
    mShadowNearPlane = header->mShadowNearPlane;
    mShadowFarPlane = header->mShadowFarPlane;
    mShadowAngle = header->mShadowAngle;
    mShadowRadius = header->mShadowRadius;
    mShadowStrength = header->mShadowStrength;
    mLightType = static_cast<LightType>(header->mLightType);
    mShadowType = static_cast<ShadowType>(header->mShadowType);
    mShadowMapResolution = static_cast<ShadowMapResolution>(header->mShadowMapResolution);

    mIsShadowMapResolutionChanged = true;
}

void Light::createTargets()
{
    Graphics::createTargets(&mTargets, mShadowMapResolution, &mIsCreated);
}

void Light::destroyTargets()
{
    Graphics::destroyTargets(&mTargets, &mIsCreated);
}

void ::Light::resizeTargets()
{
    Graphics::resizeTargets(&mTargets, mShadowMapResolution, &mIsShadowMapResolutionChanged);
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