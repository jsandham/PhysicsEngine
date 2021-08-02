#ifndef DEFERRED_RENDERER_STATE_H__
#define DEFERRED_RENDERER_STATE_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/Shader.h"

#include "Graphics.h"

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

namespace PhysicsEngine
{
struct DeferredRendererState
{
    // internal graphics camera state
    CameraUniform mCameraState;

    bool mRenderToScreen;

    //Shader mGeometryShader;
    int mGeometryShaderProgram;
    int mGeometryShaderModelLoc;
    int mGeometryShaderDiffuseTexLoc;
    int mGeometryShaderSpecTexLoc;

    //Shader mSimpleLitDeferredShader;
    int mSimpleLitDeferredShaderProgram;
    int mSimpleLitDeferredShaderViewPosLoc;
    int mSimpleLitDeferredShaderLightLocs[32];

    // quad
    GLuint mQuadVAO;
    GLuint mQuadVBO;
    //Shader mQuadShader;
};
} // namespace PhysicsEngine

#endif