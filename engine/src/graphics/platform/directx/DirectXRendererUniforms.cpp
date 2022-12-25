#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/DirectX/DirectXRendererUniforms.h"

using namespace PhysicsEngine;

DirectXCameraUniform::DirectXCameraUniform()
{
}

DirectXCameraUniform::~DirectXCameraUniform()
{
}

void DirectXCameraUniform::setProjection(const glm::mat4& projection)
{
}

void DirectXCameraUniform::setView(const glm::mat4& view)
{
}

void DirectXCameraUniform::setViewProjection(const glm::mat4 &viewProj)
{
}

void DirectXCameraUniform::setCameraPos(const glm::vec3& position)
{
}

void DirectXCameraUniform::copyToUniformsToDevice()
{

}

DirectXLightUniform::DirectXLightUniform()
{
}

DirectXLightUniform::~DirectXLightUniform()
{
}

void DirectXLightUniform::setDirLightCascadeProj(int index, const glm::mat4 &projection)
{
}

void DirectXLightUniform::setDirLightCascadeView(int index, const glm::mat4& view)
{
}

void DirectXLightUniform::setDirLightCascadeEnd(int index, float cascadeEnd)
{
}

void DirectXLightUniform::setLightPosition(const glm::vec3 &position)
{
}

void DirectXLightUniform::setLightDirection(const glm::vec3 &direction)
{
}

void DirectXLightUniform::setLightColor(const glm::vec4 &color)
{
}

void DirectXLightUniform::setLightIntensity(float intensity)
{
}

void DirectXLightUniform::setSpotLightAngle(float angle)
{
}

void DirectXLightUniform::setInnerSpotLightAngle(float innerAngle)
{
}

void DirectXLightUniform::setShadowNearPlane(float NearPlane)
{
}

void DirectXLightUniform::setShadowFarPlane(float farPlane)
{
}

void DirectXLightUniform::setShadowBias(float bias)
{
}

void DirectXLightUniform::setShadowRadius(float radius)
{
}

void DirectXLightUniform::setShadowStrength(float strength)
{
}

void DirectXLightUniform::copyToUniformsToDevice()
{
}