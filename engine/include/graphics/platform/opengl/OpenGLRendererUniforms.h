#ifndef OPENGL_RENDERER_UNIFORMS_H__
#define OPENGL_RENDERER_UNFORMS_H__

#include "../../RendererUniforms.h"

namespace PhysicsEngine
{
class OpenGLCameraUniform : public CameraUniform
{
  private:
    UniformBuffer *mBuffer;

    glm::mat4 mProjection;     // 0
    glm::mat4 mView;           // 64
    glm::mat4 mViewProjection; // 128
    glm::vec3 mCameraPos;      // 192

  public:
    OpenGLCameraUniform();
    ~OpenGLCameraUniform();

    void setProjection(const glm::mat4& projection) override;
    void setView(const glm::mat4& view) override;
    void setViewProjection(const glm::mat4 &viewProj) override;
    void setCameraPos(const glm::vec3& position) override;

    void copyToUniformsToDevice() override;
};

class OpenGLLightUniform : public LightUniform
{
  private:
    UniformBuffer *mBuffer;

    glm::mat4 mLightProjection[5]; // 0    64   128  192  256
    glm::mat4 mLightView[5];       // 320  384  448  512  576
    glm::vec3 mPosition;           // 640
    glm::vec3 mDirection;          // 656
    glm::vec4 mColor;              // 672
    float mCascadeEnds[5];         // 688  704  720  736  752
    float mIntensity;              // 768
    float mSpotAngle;              // 772
    float mInnerSpotAngle;         // 776
    float mShadowNearPlane;        // 780
    float mShadowFarPlane;         // 784
    float mShadowBias;             // 788
    float mShadowRadius;           // 792
    float mShadowStrength;         // 796

  public:
    OpenGLLightUniform();
    ~OpenGLLightUniform();

    void setDirLightCascadeProj(int index, const glm::mat4& projection) override;
    void setDirLightCascadeView(int index, const glm::mat4& view) override;
    void setDirLightCascadeEnd(int index, float cascadeEnd) override;
    void setLightPosition(const glm::vec3 &position) override;
    void setLightDirection(const glm::vec3 &direction) override;
    void setLightColor(const glm::vec4 &color) override;
    void setLightIntensity(float intensity) override;
    void setSpotLightAngle(float angle) override;
    void setInnerSpotLightAngle(float innerAngle) override;
    void setShadowNearPlane(float nearPlane) override;
    void setShadowFarPlane(float farPlane) override;
    void setShadowBias(float bias) override;
    void setShadowRadius(float radius) override;
    void setShadowStrength(float strength) override;

    void copyToUniformsToDevice() override;
};
} // namespace PhysicsEngine

#endif

