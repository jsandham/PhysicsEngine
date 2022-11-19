#include "../../../../include/graphics/platform/opengl/OpenGLRendererShaders.h"
#include "../../../../include/graphics/platform/opengl/OpenGLRenderer.h"
#include "../../../../include/core/Shader.h"
#include "../../../../include/core/Log.h"

#include "GLSL/glsl_shaders.h"

using namespace PhysicsEngine;

void OpenGLRendererShaders::init_impl()
{
    mSSAOShader.mProgram = -1;
    mGeometryShader.mProgram = -1;
    mDepthShader.mProgram = -1;
    mDepthCubemapShader.mProgram = -1;
    mScreenQuadShader.mProgram = -1;
    mSpriteShader.mProgram = -1;
    mGBufferShader.mProgram = -1;
    mColorShader.mProgram = -1;
    mColorInstancedShader.mProgram = -1;
    mNormalShader.mProgram = -1;
    mNormalInstancedShader.mProgram = -1;
    mPositionShader.mProgram = -1;
    mPositionInstancedShader.mProgram = -1;
    mLinearDepthShader.mProgram = -1;
    mLinearDepthInstancedShader.mProgram = -1;
    mLineShader.mProgram = -1;
    mGizmoShader.mProgram = -1;
    mGridShader.mProgram = -1;

    Log::warn("Start compile shaders\n");
    compileSSAOShader();
    compileGeometryShader();
    compileDepthShader();
    compileDepthCubemapShader();
    compileScreenQuadShader();
    compileSpriteShader();
    compileGBufferShader();
    compileColorShader();
    compileColorInstancedShader();
    compileNormalShader();
    compileNormalInstancedShader();
    compilePositionShader();
    compilePositionInstancedShader();
    compileLinearDepthShader();
    compileLinearDepthInstancedShader();
    compileLineShader();
    compileGizmoShader();
    compileGridShader();
    Log::warn("End compile shaders\n");
}

void OpenGLRendererShaders::compileSSAOShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("SSAO", getSSAOVertexShader(), getSSAOFragmentShader(), "", &program, status);
    if (status.mShaderLinked)
    {
        mSSAOShader.mProgram = program;
        mSSAOShader.mProjectionLoc = OpenGLRenderer::findUniformLocation("projection", mSSAOShader.mProgram);
        mSSAOShader.mPositionTexLoc = OpenGLRenderer::findUniformLocation("positionTex", mSSAOShader.mProgram);
        mSSAOShader.mNormalTexLoc = OpenGLRenderer::findUniformLocation("normalTex", mSSAOShader.mProgram);
        mSSAOShader.mNoiseTexLoc = OpenGLRenderer::findUniformLocation("noiseTex", mSSAOShader.mProgram);

        for (int i = 0; i < 64; i++)
        {
            std::string sample = "samples[" + std::to_string(i) + "]";
            mSSAOShader.mSamplesLoc[i] = OpenGLRenderer::findUniformLocation(sample.c_str(), mSSAOShader.mProgram);
        }
    }
}

void OpenGLRendererShaders::compileGeometryShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Geometry", getGeometryVertexShader(), getGeometryFragmentShader(), "", &program, status);
    if (status.mShaderLinked)
    {
        mGeometryShader.mProgram = program;
        mGeometryShader.mModelLoc = OpenGLRenderer::findUniformLocation("model", mGeometryShader.mProgram);

        OpenGLRenderer::setUniformBlock("CameraBlock", 0, mGeometryShader.mProgram);
    }
}

void OpenGLRendererShaders::compileDepthShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Shadow Depth Map", getShadowDepthMapVertexShader(), getShadowDepthMapFragmentShader(), "",
        &program, status);
    if (status.mShaderLinked)
    {
        mDepthShader.mProgram = program;
        mDepthShader.mModelLoc = OpenGLRenderer::findUniformLocation("model", mDepthShader.mProgram);
        mDepthShader.mViewLoc = OpenGLRenderer::findUniformLocation("view", mDepthShader.mProgram);
        mDepthShader.mProjectionLoc = OpenGLRenderer::findUniformLocation("projection", mDepthShader.mProgram);
    }
}

void OpenGLRendererShaders::compileDepthCubemapShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Shadow Depth Cubemap", getShadowDepthCubemapVertexShader(),
        getShadowDepthCubemapFragmentShader(), getShadowDepthCubemapGeometryShader(), &program, status);
    if (status.mShaderLinked)
    {
        mDepthCubemapShader.mProgram = program;
        mDepthCubemapShader.mLightPosLoc =
            OpenGLRenderer::findUniformLocation("lightPos", mDepthCubemapShader.mProgram);
        mDepthCubemapShader.mFarPlaneLoc =
            OpenGLRenderer::findUniformLocation("farPlane", mDepthCubemapShader.mProgram);
        mDepthCubemapShader.mModelLoc = OpenGLRenderer::findUniformLocation("model", mDepthCubemapShader.mProgram);
        mDepthCubemapShader.mCubeViewProjMatricesLoc0 =
            OpenGLRenderer::findUniformLocation("cubeViewProjMatrices[0]", mDepthCubemapShader.mProgram);
        mDepthCubemapShader.mCubeViewProjMatricesLoc1 =
            OpenGLRenderer::findUniformLocation("cubeViewProjMatrices[1]", mDepthCubemapShader.mProgram);
        mDepthCubemapShader.mCubeViewProjMatricesLoc2 =
            OpenGLRenderer::findUniformLocation("cubeViewProjMatrices[2]", mDepthCubemapShader.mProgram);
        mDepthCubemapShader.mCubeViewProjMatricesLoc3 =
            OpenGLRenderer::findUniformLocation("cubeViewProjMatrices[3]", mDepthCubemapShader.mProgram);
        mDepthCubemapShader.mCubeViewProjMatricesLoc4 =
            OpenGLRenderer::findUniformLocation("cubeViewProjMatrices[4]", mDepthCubemapShader.mProgram);
        mDepthCubemapShader.mCubeViewProjMatricesLoc5 =
            OpenGLRenderer::findUniformLocation("cubeViewProjMatrices[5]", mDepthCubemapShader.mProgram);
    }
}

void OpenGLRendererShaders::compileScreenQuadShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Screen Quad", getScreenQuadVertexShader(), getScreenQuadFragmentShader(), "", &program, status);
    if (status.mShaderLinked)
    {
        mScreenQuadShader.mProgram = program;
        mScreenQuadShader.mTexLoc = OpenGLRenderer::findUniformLocation("screenTexture", mScreenQuadShader.mProgram);
    }
}

void OpenGLRendererShaders::compileSpriteShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Sprite", getSpriteVertexShader(), getSpriteFragmentShader(), "", &program, status);
    if (status.mShaderLinked)
    {
        mSpriteShader.mProgram = program;
        mSpriteShader.mModelLoc = OpenGLRenderer::findUniformLocation("model", mSpriteShader.mProgram);
        mSpriteShader.mViewLoc = OpenGLRenderer::findUniformLocation("view", mSpriteShader.mProgram);
        mSpriteShader.mProjectionLoc = OpenGLRenderer::findUniformLocation("projection", mSpriteShader.mProgram);
        mSpriteShader.mColorLoc = OpenGLRenderer::findUniformLocation("spriteColor", mSpriteShader.mProgram);
        mSpriteShader.mImageLoc = OpenGLRenderer::findUniformLocation("image", mSpriteShader.mProgram);
    }
}

void OpenGLRendererShaders::compileGBufferShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("GBuffer", getGBufferVertexShader(), getGBufferFragmentShader(), "", &program, status);
    if (status.mShaderLinked)
    {
        mGBufferShader.mProgram = program;
        mGBufferShader.mModelLoc = OpenGLRenderer::findUniformLocation("model", mGBufferShader.mProgram);
        mGBufferShader.mDiffuseTexLoc =
            OpenGLRenderer::findUniformLocation("texture_diffuse1", mGBufferShader.mProgram);
        mGBufferShader.mSpecTexLoc =
            OpenGLRenderer::findUniformLocation("texture_specular1", mGBufferShader.mProgram);

        OpenGLRenderer::setUniformBlock("CameraBlock", 0, mGBufferShader.mProgram);
    }
}

void OpenGLRendererShaders::compileColorShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Color", getColorVertexShader(), getColorFragmentShader(), "", &program, status);
    if (status.mShaderLinked)
    {
        mColorShader.mProgram = program;
        mColorShader.mModelLoc = OpenGLRenderer::findUniformLocation("model", mColorShader.mProgram);
        mColorShader.mColorLoc = OpenGLRenderer::findUniformLocation("material.color", mColorShader.mProgram);

        OpenGLRenderer::setUniformBlock("CameraBlock", 0, mColorShader.mProgram);
    }
}

void OpenGLRendererShaders::compileColorInstancedShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Color Instanced", getColorInstancedVertexShader(), getColorInstancedFragmentShader(), "",
        &program, status);
    if (status.mShaderLinked)
    {
        mColorInstancedShader.mProgram = program;

        OpenGLRenderer::setUniformBlock("CameraBlock", 0, mColorInstancedShader.mProgram);
    }
}

void OpenGLRendererShaders::compileNormalShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Normal", getNormalVertexShader(), getNormalFragmentShader(), "", &program, status);
    if (status.mShaderLinked)
    {
        mNormalShader.mProgram = program;
        mNormalShader.mModelLoc = OpenGLRenderer::findUniformLocation("model", mNormalShader.mProgram);

        OpenGLRenderer::setUniformBlock("CameraBlock", 0, mNormalShader.mProgram);
    }
}

void OpenGLRendererShaders::compileNormalInstancedShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Normal Instanced", getNormalInstancedVertexShader(), getNormalInstancedFragmentShader(), "",
        &program, status);
    if (status.mShaderLinked)
    {
        mNormalInstancedShader.mProgram = program;

        OpenGLRenderer::setUniformBlock("CameraBlock", 0, mNormalInstancedShader.mProgram);
    }
}

void OpenGLRendererShaders::compilePositionShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Position", getPositionVertexShader(), getPositionFragmentShader(), "", &program, status);
    if (status.mShaderLinked)
    {
        mPositionShader.mProgram = program;
        mPositionShader.mModelLoc = OpenGLRenderer::findUniformLocation("model", mPositionShader.mProgram);

        OpenGLRenderer::setUniformBlock("CameraBlock", 0, mPositionShader.mProgram);
    }
}

void OpenGLRendererShaders::compilePositionInstancedShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Position Instanced", getPositionInstancedVertexShader(), getPositionInstancedFragmentShader(),
        "", &program, status);
    if (status.mShaderLinked)
    {
        mPositionInstancedShader.mProgram = program;

        OpenGLRenderer::setUniformBlock("CameraBlock", 0, mPositionInstancedShader.mProgram);
    }
}

void OpenGLRendererShaders::compileLinearDepthShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Linear Depth", getLinearDepthVertexShader(), getLinearDepthFragmentShader(), "", &program,
        status);
    if (status.mShaderLinked)
    {
        mLinearDepthShader.mProgram = program;
        mLinearDepthShader.mModelLoc = OpenGLRenderer::findUniformLocation("model", mLinearDepthShader.mProgram);

        OpenGLRenderer::setUniformBlock("CameraBlock", 0, mLinearDepthShader.mProgram);
    }
}

void OpenGLRendererShaders::compileLinearDepthInstancedShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Linear Depth Instanced", getLinearDepthInstancedVertexShader(),
        getLinearDepthInstancedFragmentShader(), "", &program, status);
    if (status.mShaderLinked)
    {
        mLinearDepthInstancedShader.mProgram = program;
        OpenGLRenderer::setUniformBlock("CameraBlock", 0, mLinearDepthInstancedShader.mProgram);
    }
}

void OpenGLRendererShaders::compileLineShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Line", getLineVertexShader(), getLineFragmentShader(), "", &program, status);
    if (status.mShaderLinked)
    {
        mLineShader.mProgram = program;
        mLineShader.mMVPLoc = OpenGLRenderer::findUniformLocation("mvp", mLineShader.mProgram);
    }
}

void OpenGLRendererShaders::compileGizmoShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Gizmo", getGizmoVertexShader(), getGizmoFragmentShader(), "", &program, status);
    if (status.mShaderLinked)
    {
        mGizmoShader.mProgram = program;
        mGizmoShader.mColorLoc = OpenGLRenderer::findUniformLocation("color", mGizmoShader.mProgram);
        mGizmoShader.mLightPosLoc = OpenGLRenderer::findUniformLocation("lightPos", mGizmoShader.mProgram);
        mGizmoShader.mModelLoc = OpenGLRenderer::findUniformLocation("model", mGizmoShader.mProgram);
        mGizmoShader.mViewLoc = OpenGLRenderer::findUniformLocation("view", mGizmoShader.mProgram);
        mGizmoShader.mProjLoc = OpenGLRenderer::findUniformLocation("projection", mGizmoShader.mProgram);
    }
}

void OpenGLRendererShaders::compileGridShader()
{
    ShaderStatus status;
    unsigned int program = 0;
    OpenGLRenderer::compile("Grid", getGridVertexShader(), getGridFragmentShader(), "", &program, status);
    if (status.mShaderLinked)
    {
        mGridShader.mProgram = program;
        mGridShader.mMVPLoc = OpenGLRenderer::findUniformLocation("mvp", mGridShader.mProgram);
        mGridShader.mColorLoc = OpenGLRenderer::findUniformLocation("color", mGridShader.mProgram);

        OpenGLRenderer::setUniformBlock("CameraBlock", 0, mGridShader.mProgram);
    }
}

SSAOShader OpenGLRendererShaders::getSSAOShader_impl()
{
	return mSSAOShader;
}

GeometryShader OpenGLRendererShaders::getGeometryShader_impl()
{
    return mGeometryShader;
}

DepthShader OpenGLRendererShaders::getDepthShader_impl()
{
	return mDepthShader;
}

DepthCubemapShader OpenGLRendererShaders::getDepthCubemapShader_impl()
{
	return mDepthCubemapShader;
}

ScreenQuadShader OpenGLRendererShaders::getScreenQuadShader_impl()
{
	return mScreenQuadShader;
}

SpriteShader OpenGLRendererShaders::getSpriteShader_impl()
{
	return mSpriteShader;
}

GBufferShader OpenGLRendererShaders::getGBufferShader_impl()
{
	return mGBufferShader;
}

ColorShader OpenGLRendererShaders::getColorShader_impl()
{
	return mColorShader;
}

ColorInstancedShader OpenGLRendererShaders::getColorInstancedShader_impl()
{
	return mColorInstancedShader;
}

NormalShader OpenGLRendererShaders::getNormalShader_impl()
{
	return mNormalShader;
}

NormalInstancedShader OpenGLRendererShaders::getNormalInstancedShader_impl()
{
	return mNormalInstancedShader;
}

PositionShader OpenGLRendererShaders::getPositionShader_impl()
{
	return mPositionShader;
}

PositionInstancedShader OpenGLRendererShaders::getPositionInstancedShader_impl()
{
	return mPositionInstancedShader;
}

LinearDepthShader OpenGLRendererShaders::getLinearDepthShader_impl()
{
	return mLinearDepthShader;
}

LinearDepthInstancedShader OpenGLRendererShaders::getLinearDepthInstancedShader_impl()
{
	return mLinearDepthInstancedShader;
}

LineShader OpenGLRendererShaders::getLineShader_impl()
{
	return mLineShader;
}

GizmoShader OpenGLRendererShaders::getGizmoShader_impl()
{
	return mGizmoShader;
}

GridShader OpenGLRendererShaders::getGridShader_impl()
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