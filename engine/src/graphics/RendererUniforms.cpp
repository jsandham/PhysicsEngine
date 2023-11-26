#include "../../include/core/Log.h"

#include "../../include/graphics/RendererUniforms.h"

#include <glm/gtc/type_ptr.hpp>

#define GLM_FORCE_RADIANS

using namespace PhysicsEngine;

CameraUniform::CameraUniform()
{
    mBuffer = UniformBuffer::create(208, 0);

    mProjection = glm::mat4(1.0f);
    mView = glm::mat4(1.0f);
    mViewProjection = glm::mat4(1.0f);
    mCameraPos = glm::vec3(0.0f, 0.0f, 0.0f);
}

CameraUniform::~CameraUniform()
{
    delete mBuffer;
}

void CameraUniform::setProjection(const glm::mat4 &projection)
{
    mProjection = projection;
}

void CameraUniform::setView(const glm::mat4 &view)
{
    mView = view;
}

void CameraUniform::setViewProjection(const glm::mat4 &viewProj)
{
    mViewProjection = viewProj;
}

void CameraUniform::setCameraPos(const glm::vec3 &position)
{
    mCameraPos = position;
}

void CameraUniform::copyToUniformsToDevice()
{
    mBuffer->setData(glm::value_ptr(mProjection), 0, 64);
    mBuffer->setData(glm::value_ptr(mView), 64, 64);
    mBuffer->setData(glm::value_ptr(mViewProjection), 128, 64);
    mBuffer->setData(glm::value_ptr(mCameraPos), 192, 12);

    mBuffer->bind(PipelineStage::VS);
    mBuffer->bind(PipelineStage::PS);
    mBuffer->copyDataToDevice();
    mBuffer->unbind(PipelineStage::VS);
    mBuffer->unbind(PipelineStage::PS);
}

LightUniform::LightUniform()
{
    mBuffer = UniformBuffer::create(800, 1);

    for (int i = 0; i < 5; i++)
    {
        mLightProjection[i] = glm::mat4(1.0f);
        mLightView[i] = glm::mat4(1.0f);
    }

    mCascadeEnds[0] = 5.0f;
    mCascadeEnds[1] = 10.0f;
    mCascadeEnds[2] = 20.0f;
    mCascadeEnds[3] = 40.0f;
    mCascadeEnds[4] = 250.0f;

    mPosition = glm::vec3(0.0f, 10.0f, 0.0f);
    mDirection = glm::vec3(0.0, -1.0f, 0.0f);
    mColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

    mIntensity = 1.0f;
    mSpotAngle = 15.0f;      // glm::cos(glm::radians(15.0f));
    mInnerSpotAngle = 12.5f; // glm::cos(glm::radians(12.5f));
    mShadowNearPlane = 0.1f;
    mShadowFarPlane = 10.0f;
    mShadowBias = 0.005f;
    mShadowRadius = 0.0f;
    mShadowStrength = 1.0f;
}

LightUniform::~LightUniform()
{
    delete mBuffer;
}

void LightUniform::setDirLightCascadeProj(int index, const glm::mat4 &projection)
{
    assert(index >= 0 && index <= 5);

    mLightProjection[index] = projection;
}

void LightUniform::setDirLightCascadeView(int index, const glm::mat4 &view)
{
    assert(index >= 0 && index <= 5);

    mLightView[index] = view;
}

void LightUniform::setDirLightCascadeEnd(int index, float cascadeEnd)
{
    assert(index >= 0 && index <= 5);

    mCascadeEnds[index] = cascadeEnd;
}

void LightUniform::setLightPosition(const glm::vec3 &position)
{
    mPosition = position;
}

void LightUniform::setLightDirection(const glm::vec3 &direction)
{
    mDirection = direction;
}

void LightUniform::setLightColor(const glm::vec4 &color)
{
    mColor = color;
}

void LightUniform::setLightIntensity(float intensity)
{
    mIntensity = intensity;
}

void LightUniform::setSpotLightAngle(float angle)
{
    mSpotAngle = angle;
}

void LightUniform::setInnerSpotLightAngle(float innerAngle)
{
    mInnerSpotAngle = innerAngle;
}

void LightUniform::setShadowNearPlane(float nearPlane)
{
    mShadowNearPlane = nearPlane;
}

void LightUniform::setShadowFarPlane(float farPlane)
{
    mShadowFarPlane = farPlane;
}

void LightUniform::setShadowBias(float bias)
{
    mShadowBias = bias;
}

void LightUniform::setShadowRadius(float radius)
{
    mShadowRadius = radius;
}

void LightUniform::setShadowStrength(float strength)
{
    mShadowStrength = strength;
}

void LightUniform::copyToUniformsToDevice()
{
    mBuffer->setData(&mLightProjection[0], 0, 320);
    mBuffer->setData(&mLightView[0], 320, 320);
    mBuffer->setData(glm::value_ptr(mPosition), 640, 12);
    mBuffer->setData(glm::value_ptr(mDirection), 656, 12);
    mBuffer->setData(glm::value_ptr(mColor), 672, 16);
    mBuffer->setData(&mCascadeEnds[0], 688, 4);
    mBuffer->setData(&mCascadeEnds[1], 704, 4);
    mBuffer->setData(&mCascadeEnds[2], 720, 4);
    mBuffer->setData(&mCascadeEnds[3], 736, 4);
    mBuffer->setData(&mCascadeEnds[4], 752, 4);
    mBuffer->setData(&mIntensity, 768, 4);

    float spotAngle = glm::cos(glm::radians(mSpotAngle));
    float innerSpotAngle = glm::cos(glm::radians(mInnerSpotAngle));
    mBuffer->setData(&spotAngle, 772, 4);
    mBuffer->setData(&innerSpotAngle, 776, 4);
    mBuffer->setData(&mShadowNearPlane, 780, 4);
    mBuffer->setData(&mShadowFarPlane, 784, 4);
    mBuffer->setData(&mShadowBias, 788, 4);
    mBuffer->setData(&mShadowRadius, 792, 4);
    mBuffer->setData(&mShadowStrength, 796, 4);

    mBuffer->bind(PipelineStage::VS);
    mBuffer->bind(PipelineStage::PS);
    mBuffer->copyDataToDevice();
    mBuffer->unbind(PipelineStage::VS);
    mBuffer->unbind(PipelineStage::PS);
}

OcclusionUniform::OcclusionUniform()
{
    mBuffer = UniformBuffer::create(64 * 20, 2);
}

OcclusionUniform::~OcclusionUniform()
{
    delete mBuffer;
}

void OcclusionUniform::setModel(const glm::mat4 &model, int index)
{
    assert(index >= 0);
    assert(index < 20);
    mModels[index] = model;
}

void OcclusionUniform::copyToUniformsToDevice()
{
    mBuffer->setData(glm::value_ptr(mModels[0]), 0, 20 * 64);

    mBuffer->bind(PipelineStage::VS);
    mBuffer->bind(PipelineStage::PS);
    mBuffer->copyDataToDevice();
    mBuffer->unbind(PipelineStage::VS);
    mBuffer->unbind(PipelineStage::PS);
}

CameraUniform *RendererUniforms::sCameraUniform = nullptr;
LightUniform *RendererUniforms::sLightUniform = nullptr;
OcclusionUniform *RendererUniforms::sOcclusionUniform = nullptr;

void RendererUniforms::createInternalUniforms()
{
    // Note these pointers never free'd but they are static and
    // exist for the length of the program so ... meh?
    sCameraUniform = new CameraUniform();
    sLightUniform = new LightUniform();
    sOcclusionUniform = new OcclusionUniform();
}

CameraUniform *RendererUniforms::getCameraUniform()
{
    return RendererUniforms::sCameraUniform;
}

LightUniform *RendererUniforms::getLightUniform()
{
    return RendererUniforms::sLightUniform;
}

OcclusionUniform *RendererUniforms::getOcclusionUniform()
{
    return RendererUniforms::sOcclusionUniform;
}