#ifndef __GRAPHICSTATE_H__
#define __GRAPHICSTATE_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "../glm/glm.hpp"
#include "../glm/gtc/type_ptr.hpp"

namespace PhysicsEngine
{
struct GraphicsCameraState
{
    glm::mat4 mProjection; // 0
    glm::mat4 mView;       // 64
    glm::vec3 mCameraPos;  // 128

    GLuint mHandle;
};

struct GraphicsLightState
{
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
    float mShadowAngle;            // 788
    float mShadowRadius;           // 792
    float mShadowStrength;         // 796

    GLuint mHandle;
};
} // namespace PhysicsEngine

#endif