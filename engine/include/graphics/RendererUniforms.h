#ifndef RENDERER_UNIFORMS_H__
#define RENDERER_UNIFORMS_H__

#include "UniformBuffer.h"
#include <string>
#include <glm/glm.hpp>

namespace PhysicsEngine
{
class CameraUniform
{
  public:
    CameraUniform(){};
    CameraUniform(const CameraUniform &other) = delete;
    CameraUniform &operator=(const CameraUniform &other) = delete;
    virtual ~CameraUniform(){};

    virtual void setProjection(const glm::mat4& projection) = 0;
    virtual void setView(const glm::mat4& view) = 0;
    virtual void setViewProjection(const glm::mat4& viewProj) = 0;
    virtual void setCameraPos(const glm::vec3& position) = 0;

    virtual void copyToUniformsToDevice() = 0;

    static CameraUniform *create();
};

class LightUniform
{
  public:
    LightUniform(){};
    LightUniform(const LightUniform &other) = delete;
    LightUniform &operator=(const LightUniform &other) = delete;
    virtual ~LightUniform(){};

    virtual void setDirLightCascadeProj(int index, const glm::mat4& projection) = 0;
    virtual void setDirLightCascadeView(int index, const glm::mat4 &view) = 0;
    virtual void setDirLightCascadeEnd(int index, float cascadeEnd) = 0;
    virtual void setLightPosition(const glm::vec3& position) = 0;
    virtual void setLightDirection(const glm::vec3 &direction) = 0;
    virtual void setLightColor(const glm::vec4 &color) = 0;
    virtual void setLightIntensity(float intensity) = 0;
    virtual void setSpotLightAngle(float angle) = 0;
    virtual void setInnerSpotLightAngle(float innerAngle) = 0;
    virtual void setShadowNearPlane(float NearPlane) = 0;
    virtual void setShadowFarPlane(float farPlane) = 0;
    virtual void setShadowBias(float bias) = 0;
    virtual void setShadowRadius(float radius) = 0;
    virtual void setShadowStrength(float strength) = 0;

    virtual void copyToUniformsToDevice() = 0;

    static LightUniform *create();
};

class RendererUniforms
{
  private:
    static CameraUniform *sCameraUniform;
    static LightUniform *sLightUniform;

  public:
    static CameraUniform *getCameraUniform();
    static LightUniform *getLightUniform();

    static void createInternalUniforms();
};
} // namespace PhysicsEngine

#endif