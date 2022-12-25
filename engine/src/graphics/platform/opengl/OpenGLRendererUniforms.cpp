#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/opengl/OpenGLRendererUniforms.h"

#include <glm/gtc/type_ptr.hpp>

#define GLM_FORCE_RADIANS

using namespace PhysicsEngine;

OpenGLCameraUniform::OpenGLCameraUniform()
{
    mBuffer = UniformBuffer::create(204, 0);
}

OpenGLCameraUniform::~OpenGLCameraUniform()
{
    delete mBuffer;
}

void OpenGLCameraUniform::setProjection(const glm::mat4& projection)
{
    mProjection = projection;
}

void OpenGLCameraUniform::setView(const glm::mat4& view)
{
    mView = view;
}

void OpenGLCameraUniform::setViewProjection(const glm::mat4 &viewProj)
{
    mViewProjection = viewProj;
}

void OpenGLCameraUniform::setCameraPos(const glm::vec3& position)
{
    mCameraPos = position;
}

void OpenGLCameraUniform::copyToUniformsToDevice()
{
    mBuffer->bind();
    mBuffer->setData(glm::value_ptr(mProjection),0, 64);
    mBuffer->setData(glm::value_ptr(mView), 64, 64);
    mBuffer->setData(glm::value_ptr(mViewProjection), 128, 64);
    mBuffer->setData(glm::value_ptr(mCameraPos), 192, 12);
    mBuffer->unbind();
}

OpenGLLightUniform::OpenGLLightUniform()
{
    mBuffer = UniformBuffer::create(824, 1);
}

OpenGLLightUniform::~OpenGLLightUniform()
{
    delete mBuffer;
}

void OpenGLLightUniform::setDirLightCascadeProj(int index, const glm::mat4 &projection)
{
    assert(index >= 0 && index <= 5);

    mLightProjection[index] = projection;
}

void OpenGLLightUniform::setDirLightCascadeView(int index, const glm::mat4& view)
{
    assert(index >= 0 && index <= 5);

    mLightView[index] = view;
}

void OpenGLLightUniform::setDirLightCascadeEnd(int index, float cascadeEnd)
{
    assert(index >= 0 && index <= 5);

    mCascadeEnds[index] = cascadeEnd;
}

void OpenGLLightUniform::setLightPosition(const glm::vec3 &position)
{
    mPosition = position;
}

void OpenGLLightUniform::setLightDirection(const glm::vec3 &direction)
{
    mDirection = direction;
}

void OpenGLLightUniform::setLightColor(const glm::vec4 &color)
{
    mColor = color;
}

void OpenGLLightUniform::setLightIntensity(float intensity)
{
    mIntensity = intensity;
}

void OpenGLLightUniform::setSpotLightAngle(float angle)
{
    mSpotAngle = angle;
}

void OpenGLLightUniform::setInnerSpotLightAngle(float innerAngle)
{
    mInnerSpotAngle = innerAngle;
}

void OpenGLLightUniform::setShadowNearPlane(float nearPlane)
{
    mShadowNearPlane = nearPlane;
}

void OpenGLLightUniform::setShadowFarPlane(float farPlane)
{
    mShadowFarPlane = farPlane;
}

void OpenGLLightUniform::setShadowBias(float bias)
{
    mShadowBias = bias;
}

void OpenGLLightUniform::setShadowRadius(float radius)
{
    mShadowRadius = radius;
}

void OpenGLLightUniform::setShadowStrength(float strength)
{
    mShadowStrength = strength;
}

void OpenGLLightUniform::copyToUniformsToDevice()
{
    mBuffer->bind();
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
    mBuffer->unbind();
}
