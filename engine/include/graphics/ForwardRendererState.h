#ifndef FORWARD_RENDERER_STATE_H__
#define FORWARD_RENDERER_STATE_H__

#include "../core/Shader.h"

#include "Graphics.h"

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

namespace PhysicsEngine
{
struct ForwardRendererState
{
    // internal graphics camera state
    CameraUniform mCameraState;

    // internal graphics light state
    LightUniform mLightState;

    bool mRenderToScreen;

    // directional light cascade shadow map data
    float mCascadeEnds[6];
    glm::mat4 mCascadeOrthoProj[5];
    glm::mat4 mCascadeLightView[5];

    int mDepthShaderProgram;
    int mDepthShaderModelLoc;
    int mDepthShaderViewLoc;
    int mDepthShaderProjectionLoc;

    // spotlight shadow map data
    glm::mat4 mShadowViewMatrix;
    glm::mat4 mShadowProjMatrix;

    // pointlight cubemap shadow map data
    glm::mat4 mCubeViewProjMatrices[6];

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

    int mGeometryShaderProgram;
    int mGeometryShaderModelLoc;

    int mColorShaderProgram;
    int mColorShaderModelLoc;
    int mColorShaderColorLoc;

    // ssao
    int mSsaoShaderProgram;
    int mSsaoShaderProjectionLoc;
    int mSsaoShaderPositionTexLoc;
    int mSsaoShaderNormalTexLoc;
    int mSsaoShaderNoiseTexLoc;
    int mSsaoShaderSamplesLoc[64];

    // sprite
    int mSpriteShaderProgram;
    int mSpriteModelLoc;
    int mSpriteViewLoc;
    int mSpriteProjectionLoc;
    int mSpriteColorLoc;
    int mSpriteImageLoc;

    // quad
    unsigned int mQuadVAO;
    unsigned int mQuadVBO;
};
} // namespace PhysicsEngine

#endif