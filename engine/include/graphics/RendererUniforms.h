#ifndef RENDERER_UNIFORMS_H__
#define RENDERER_UNIFORMS_H__

#include "UniformBuffer.h"
#include <glm/glm.hpp>
#include <string>

namespace PhysicsEngine
{
class CameraUniform
{
  private:
    UniformBuffer *mBuffer;

    glm::mat4 mProjection;     // 0
    glm::mat4 mView;           // 64
    glm::mat4 mViewProjection; // 128
    glm::vec3 mCameraPos;      // 192

  public:
    CameraUniform();
    ~CameraUniform();

    void setProjection(const glm::mat4 &projection);
    void setView(const glm::mat4 &view);
    void setViewProjection(const glm::mat4 &viewProj);
    void setCameraPos(const glm::vec3 &position);

    void copyToUniformsToDevice();
};

class LightUniform
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
    LightUniform();
    ~LightUniform();

    void setDirLightCascadeProj(int index, const glm::mat4 &projection);
    void setDirLightCascadeView(int index, const glm::mat4 &view);
    void setDirLightCascadeEnd(int index, float cascadeEnd);
    void setLightPosition(const glm::vec3 &position);
    void setLightDirection(const glm::vec3 &direction);
    void setLightColor(const glm::vec4 &color);
    void setLightIntensity(float intensity);
    void setSpotLightAngle(float angle);
    void setInnerSpotLightAngle(float innerAngle);
    void setShadowNearPlane(float nearPlane);
    void setShadowFarPlane(float farPlane);
    void setShadowBias(float bias);
    void setShadowRadius(float radius);
    void setShadowStrength(float strength);

    void copyToUniformsToDevice();
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