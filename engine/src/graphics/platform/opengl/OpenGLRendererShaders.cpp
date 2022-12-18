#include "../../../../include/graphics/platform/opengl/OpenGLRendererShaders.h"
#include "../../../../include/graphics/platform/opengl/OpenGLRenderer.h"
#include "../../../../include/core/Shader.h"
#include "../../../../include/core/Log.h"

#include "GLSL/glsl_shaders.h"

using namespace PhysicsEngine;

OpenGLRendererShaders::OpenGLRendererShaders()
{
    mSSAOShader = ShaderProgram::create();
    mGeometryShader = ShaderProgram::create();
    mDepthShader = ShaderProgram::create();
    mDepthCubemapShader = ShaderProgram::create();
    mScreenQuadShader = ShaderProgram::create();
    mSpriteShader = ShaderProgram::create();
    mGBufferShader = ShaderProgram::create();
    mColorShader = ShaderProgram::create();
    mColorInstancedShader = ShaderProgram::create();
    mNormalShader = ShaderProgram::create();
    mNormalInstancedShader = ShaderProgram::create();
    mPositionShader = ShaderProgram::create();
    mPositionInstancedShader = ShaderProgram::create();
    mLinearDepthShader = ShaderProgram::create();
    mLinearDepthInstancedShader = ShaderProgram::create();
    mLineShader = ShaderProgram::create();
    mGizmoShader = ShaderProgram::create();
    mGridShader = ShaderProgram::create();

    mSSAOShader->load("SSAO", getSSAOVertexShader(), getSSAOFragmentShader());
    mGeometryShader->load("Geometry", getGeometryVertexShader(), getGeometryFragmentShader());
    mDepthShader->load("DepthMap", getShadowDepthMapVertexShader(), getShadowDepthMapFragmentShader());
    mDepthCubemapShader->load("DepthCubemap", getShadowDepthCubemapVertexShader(), getShadowDepthCubemapFragmentShader(),
                              getShadowDepthCubemapGeometryShader());
    mScreenQuadShader->load("ScreenQuad", getScreenQuadVertexShader(), getScreenQuadFragmentShader());
    mSpriteShader->load("Sprite", getSpriteVertexShader(), getSpriteFragmentShader());
    mGBufferShader->load("GBuffer", getGBufferVertexShader(), getGBufferFragmentShader());
    mColorShader->load("Color", getColorVertexShader(), getColorFragmentShader());
    mColorInstancedShader->load("ColorInstanced", getColorInstancedVertexShader(), getColorInstancedFragmentShader());
    mNormalShader->load("Normal", getNormalVertexShader(), getNormalFragmentShader());
    mNormalInstancedShader->load("NormalInstanced", getNormalInstancedVertexShader(), getNormalInstancedFragmentShader());
    mPositionShader->load("Position", getPositionVertexShader(), getPositionFragmentShader());
    mPositionInstancedShader->load("PositionInstanced", getPositionInstancedVertexShader(), getPositionInstancedFragmentShader());
    mLinearDepthShader->load("LinearDepth", getLinearDepthVertexShader(), getLinearDepthFragmentShader());
    mLinearDepthInstancedShader->load("LinearDepthInstanced", getLinearDepthInstancedVertexShader(),
                                      getLinearDepthInstancedFragmentShader());
    mLineShader->load("Line", getLineVertexShader(), getLineFragmentShader());
    mGizmoShader->load("Gizmo", getGizmoVertexShader(), getGizmoFragmentShader());
    mGridShader->load("Grid", getGridVertexShader(), getGridFragmentShader());

    Log::warn("Start compile internal shaders\n");
    mSSAOShader->compile();
    mGeometryShader->compile();
    mDepthShader->compile();
    mDepthCubemapShader->compile();
    mScreenQuadShader->compile();
    mSpriteShader->compile();
    mGBufferShader->compile();
    mColorShader->compile();
    mColorInstancedShader->compile();
    mNormalShader->compile();
    mNormalInstancedShader->compile();
    mPositionShader->compile();
    mPositionInstancedShader->compile();
    mLinearDepthShader->compile();
    mLinearDepthInstancedShader->compile();
    mLineShader->compile();
    mGizmoShader->compile();
    mGridShader->compile();
    Log::warn("End compile internal shaders\n");
}

OpenGLRendererShaders::~OpenGLRendererShaders()
{
    delete mSSAOShader;
    delete mGeometryShader;
    delete mDepthShader;
    delete mDepthCubemapShader;
    delete mScreenQuadShader;
    delete mSpriteShader;
    delete mGBufferShader;
    delete mColorShader;
    delete mColorInstancedShader;
    delete mNormalShader;
    delete mNormalInstancedShader;
    delete mPositionShader;
    delete mPositionInstancedShader;
    delete mLinearDepthShader;
    delete mLinearDepthInstancedShader;
    delete mLineShader;
    delete mGizmoShader;
    delete mGridShader;
}

ShaderProgram *OpenGLRendererShaders::getSSAOShader_impl()
{
	return mSSAOShader;
}

ShaderProgram *OpenGLRendererShaders::getGeometryShader_impl()
{
    return mGeometryShader;
}

ShaderProgram *OpenGLRendererShaders::getDepthShader_impl()
{
	return mDepthShader;
}

ShaderProgram *OpenGLRendererShaders::getDepthCubemapShader_impl()
{
	return mDepthCubemapShader;
}

ShaderProgram *OpenGLRendererShaders::getScreenQuadShader_impl()
{
	return mScreenQuadShader;
}

ShaderProgram *OpenGLRendererShaders::getSpriteShader_impl()
{
	return mSpriteShader;
}

ShaderProgram *OpenGLRendererShaders::getGBufferShader_impl()
{
	return mGBufferShader;
}

ShaderProgram *OpenGLRendererShaders::getColorShader_impl()
{
	return mColorShader;
}

ShaderProgram *OpenGLRendererShaders::getColorInstancedShader_impl()
{
	return mColorInstancedShader;
}

ShaderProgram *OpenGLRendererShaders::getNormalShader_impl()
{
	return mNormalShader;
}

ShaderProgram *OpenGLRendererShaders::getNormalInstancedShader_impl()
{
	return mNormalInstancedShader;
}

ShaderProgram *OpenGLRendererShaders::getPositionShader_impl()
{
	return mPositionShader;
}

ShaderProgram *OpenGLRendererShaders::getPositionInstancedShader_impl()
{
	return mPositionInstancedShader;
}

ShaderProgram *OpenGLRendererShaders::getLinearDepthShader_impl()
{
	return mLinearDepthShader;
}

ShaderProgram *OpenGLRendererShaders::getLinearDepthInstancedShader_impl()
{
	return mLinearDepthInstancedShader;
}

ShaderProgram *OpenGLRendererShaders::getLineShader_impl()
{
	return mLineShader;
}

ShaderProgram *OpenGLRendererShaders::getGizmoShader_impl()
{
	return mGizmoShader;
}

ShaderProgram *OpenGLRendererShaders::getGridShader_impl()
{
	return mGridShader;
}

std::string OpenGLRendererShaders::getStandardVertexShader_impl()
{
    return PhysicsEngine::getStandardVertexShader();
}

std::string OpenGLRendererShaders::getStandardFragmentShader_impl()
{
    return PhysicsEngine::getStandardFragmentShader();
}