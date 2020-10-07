#ifndef __FORWARD_RENDERER_STATE_H__
#define __FORWARD_RENDERER_STATE_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/Shader.h"

#include "GraphicsState.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
struct ForwardRendererState
{
    // internal graphics camera state
    GraphicsCameraState mCameraState;

    // internal graphics light state
    GraphicsLightState mLightState;

    bool mRenderToScreen;

    // directional light cascade shadow map data
    // GLuint mShadowCascadeFBO[5];
    // GLuint mShadowCascadeDepth[5];
    float mCascadeEnds[6];
    glm::mat4 mCascadeOrthoProj[5];
    glm::mat4 mCascadeLightView[5];

    Shader *mDepthShader;
    int mDepthShaderProgram;
    int mDepthShaderModelLoc;
    int mDepthShaderViewLoc;
    int mDepthShaderProjectionLoc;

    // spotlight shadow map data
    // GLuint mShadowSpotlightFBO;
    // GLuint mShadowSpotlightDepth;
    glm::mat4 mShadowViewMatrix;
    glm::mat4 mShadowProjMatrix;

    // pointlight cubemap shadow map data
    // GLuint mShadowCubemapFBO;
    // GLuint mShadowCubemapDepth;
    glm::mat4 mCubeViewProjMatrices[6];

    Shader *mDepthCubemapShader;
    int mDepthCubemapShaderProgram;
    int mDepthCubemapShaderLightPosLoc;
    int mDepthCubemapShaderFarPlaneLoc;
    int mDepthCubemapShaderModelLoc;
    int mDepthCubemapShaderCubeViewProjMatricesLoc0;
    int mDepthCubemapShaderCubeViewProjMatricesLoc1;
    int mDepthCubemapShaderCubeViewProjMatricesLoc2;
    int mDepthCubemapShaderCubeViewProjMatricesLoc3;
    int mDepthCubemapShaderCubeViewProjMatricesLoc4;
    int mDepthCubemapShaderCubeViewProjMatricesLoc5;

    Shader *mGeometryShader;
    int mGeometryShaderProgram;
    int mGeometryShaderModelLoc;

    Shader *mColorShader;
    int mColorShaderProgram;
    int mColorShaderModelLoc;
    int mColorShaderColorLoc;

    // ssao
    Shader *mSsaoShader;
    int mSsaoShaderProgram;
    int mSsaoShaderProjectionLoc;
    int mSsaoShaderPositionTexLoc;
    int mSsaoShaderNormalTexLoc;
    int mSsaoShaderNoiseTexLoc;
    int mSsaoShaderSamplesLoc[64];

    // quad
    GLuint mQuadVAO;
    GLuint mQuadVBO;
    Shader *mQuadShader;
};
} // namespace PhysicsEngine

#endif