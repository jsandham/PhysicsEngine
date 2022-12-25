#ifndef DIRECTX_RENDERER_UNIFORMS_H__
#define DIRECTX_RENDERER_UNFORMS_H__

#include "../../RendererUniforms.h"

namespace PhysicsEngine
{
class DirectXCameraUniform : public CameraUniform
{
  private:
    UniformBuffer *mBuffer;

  public:
    DirectXCameraUniform();
    ~DirectXCameraUniform();

    void setProjection(const glm::mat4& projection) override;
    void setView(const glm::mat4& view) override;
    void setViewProjection(const glm::mat4& viewProj) override;
    void setCameraPos(const glm::vec3& position) override;

    void copyToUniformsToDevice() override;
};

class DirectXLightUniform : public LightUniform
{
  private:
    UniformBuffer *mBuffer;

  public:
    DirectXLightUniform();
    ~DirectXLightUniform();

    void setDirLightCascadeProj(int index, const glm::mat4 &projection) override;
    void setDirLightCascadeView(int index, const glm::mat4& view) override;
    void setDirLightCascadeEnd(int index, float cascadeEnd) override;
    void setLightPosition(const glm::vec3 &position) override;
    void setLightDirection(const glm::vec3 &direction) override;
    void setLightColor(const glm::vec4 &color) override;
    void setLightIntensity(float intensity) override;
    void setSpotLightAngle(float angle) override;
    void setInnerSpotLightAngle(float innerAngle) override;
    void setShadowNearPlane(float NearPlane) override;
    void setShadowFarPlane(float farPlane) override;
    void setShadowBias(float bias) override;
    void setShadowRadius(float radius) override;
    void setShadowStrength(float strength) override;

    void copyToUniformsToDevice() override;
};
} // namespace PhysicsEngine

#endif
