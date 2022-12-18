#include "../../../../include/graphics/platform/directx/DirectXRendererShaders.h"

using namespace PhysicsEngine;

DirectXRendererShaders::DirectXRendererShaders()
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

    /*mSSAOShader->load(getSSAOVertexShader(), getSSAOFragmentShader());
    mGeometryShader->load(getGeometryVertexShader(), getGeometryFragmentShader());
    mDepthShader->load(getShadowDepthMapVertexShader(), getShadowDepthMapFragmentShader());
    mDepthCubemapShader->load(getShadowDepthCubemapVertexShader(), getShadowDepthCubemapFragmentShader(),
                              getShadowDepthCubemapGeometryShader());
    mScreenQuadShader->load(getScreenQuadVertexShader(), getScreenQuadFragmentShader());
    mSpriteShader->load(getSpriteVertexShader(), getSpriteFragmentShader());
    mGBufferShader->load(getGBufferVertexShader(), getGBufferFragmentShader());
    mColorShader->load(getColorVertexShader(), getColorFragmentShader());
    mColorInstancedShader->load(getColorInstancedVertexShader(), getColorInstancedFragmentShader());
    mNormalShader->load(getNormalVertexShader(), getNormalFragmentShader());
    mNormalInstancedShader->load(getNormalInstancedVertexShader(), getNormalInstancedFragmentShader());
    mPositionShader->load(getPositionVertexShader(), getPositionFragmentShader());
    mPositionInstancedShader->load(getPositionInstancedVertexShader(), getPositionInstancedFragmentShader());
    mLinearDepthShader->load(getLinearDepthVertexShader(), getLinearDepthFragmentShader());
    mLinearDepthInstancedShader->load(getLinearDepthInstancedVertexShader(), getLinearDepthInstancedFragmentShader());
    mLineShader->load(getLineVertexShader(), getLineFragmentShader());
    mGizmoShader->load(getGizmoVertexShader(), getGizmoFragmentShader());
    mGridShader->load(getGridVertexShader(), getGridFragmentShader());

    Log::warn("Start compile shaders\n");
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
    Log::warn("End compile shaders\n");*/
}

DirectXRendererShaders::~DirectXRendererShaders()
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

ShaderProgram *DirectXRendererShaders::getSSAOShader_impl()
{
	return mSSAOShader;
}

ShaderProgram *DirectXRendererShaders::getGeometryShader_impl()
{
	return mGeometryShader;
}

ShaderProgram *DirectXRendererShaders::getDepthShader_impl()
{
	return mDepthShader;
}

ShaderProgram *DirectXRendererShaders::getDepthCubemapShader_impl()
{
	return mDepthCubemapShader;
}

ShaderProgram *DirectXRendererShaders::getScreenQuadShader_impl()
{
	return mScreenQuadShader;
}

ShaderProgram *DirectXRendererShaders::getSpriteShader_impl()
{
	return mSpriteShader;
}

ShaderProgram *DirectXRendererShaders::getGBufferShader_impl()
{
	return mGBufferShader;
}

ShaderProgram *DirectXRendererShaders::getColorShader_impl()
{
	return mColorShader;
}

ShaderProgram *DirectXRendererShaders::getColorInstancedShader_impl()
{
	return mColorInstancedShader;
}

ShaderProgram *DirectXRendererShaders::getNormalShader_impl()
{
	return mNormalShader;
}

ShaderProgram *DirectXRendererShaders::getNormalInstancedShader_impl()
{
	return mNormalInstancedShader;
}

ShaderProgram *DirectXRendererShaders::getPositionShader_impl()
{
	return mPositionShader;
}

ShaderProgram *DirectXRendererShaders::getPositionInstancedShader_impl()
{
	return mPositionInstancedShader;
}

ShaderProgram *DirectXRendererShaders::getLinearDepthShader_impl()
{
	return mLinearDepthShader;
}

ShaderProgram *DirectXRendererShaders::getLinearDepthInstancedShader_impl()
{
	return mLinearDepthInstancedShader;
}

ShaderProgram *DirectXRendererShaders::getLineShader_impl()
{
	return mLineShader;
}

ShaderProgram *DirectXRendererShaders::getGizmoShader_impl()
{
	return mGizmoShader;
}

ShaderProgram *DirectXRendererShaders::getGridShader_impl()
{
	return mGridShader;
}

std::string DirectXRendererShaders::getStandardVertexShader_impl()
{
	return "";
}

std::string DirectXRendererShaders::getStandardFragmentShader_impl()
{
	return "";
}