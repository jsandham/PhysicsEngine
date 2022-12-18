#include "../../include/graphics/RendererShaders.h"
#include "../../include/graphics/RenderContext.h"

#include "../../include/graphics/platform/opengl/OpenGLRendererShaders.h"
#include "../../include/graphics/platform/directx/DirectXRendererShaders.h"

using namespace PhysicsEngine;

RendererShaders* RendererShaders::sInstance = nullptr;

RendererShaders::RendererShaders()
{

}

RendererShaders::~RendererShaders()
{
}

void RendererShaders::init()
{
	switch (RenderContext::getRenderAPI())
	{
	case RenderAPI::OpenGL:
		sInstance = new OpenGLRendererShaders();
		break;
	case RenderAPI::DirectX:
		sInstance = new DirectXRendererShaders();
		break;
	}
}

RendererShaders* RendererShaders::getRendererShaders()
{
	return sInstance;
}

ShaderProgram *RendererShaders::getSSAOShader()
{
	return sInstance->getSSAOShader_impl();
}

ShaderProgram *RendererShaders::getGeometryShader()
{
	return sInstance->getGeometryShader_impl();
}

ShaderProgram *RendererShaders::getDepthShader()
{
	return sInstance->getDepthShader_impl();
}

ShaderProgram *RendererShaders::getDepthCubemapShader()
{
	return sInstance->getDepthCubemapShader_impl();
}

ShaderProgram *RendererShaders::getScreenQuadShader()
{
	return sInstance->getScreenQuadShader_impl();
}

ShaderProgram *RendererShaders::getSpriteShader()
{
	return sInstance->getSpriteShader_impl();
}

ShaderProgram *RendererShaders::getGBufferShader()
{
	return sInstance->getGBufferShader_impl();
}

ShaderProgram *RendererShaders::getColorShader()
{
	return sInstance->getColorShader_impl();
}

ShaderProgram *RendererShaders::getColorInstancedShader()
{
	return sInstance->getColorInstancedShader_impl();
}

ShaderProgram *RendererShaders::getNormalShader()
{
	return sInstance->getNormalShader_impl();
}

ShaderProgram *RendererShaders::getNormalInstancedShader()
{
	return sInstance->getNormalInstancedShader_impl();
}

ShaderProgram *RendererShaders::getPositionShader()
{
	return sInstance->getPositionShader_impl();
}

ShaderProgram *RendererShaders::getPositionInstancedShader()
{
	return sInstance->getPositionInstancedShader_impl();
}

ShaderProgram *RendererShaders::getLinearDepthShader()
{
	return sInstance->getLinearDepthShader_impl();
}

ShaderProgram *RendererShaders::getLinearDepthInstancedShader()
{
	return sInstance->getLinearDepthInstancedShader_impl();
}

ShaderProgram *RendererShaders::getLineShader()
{
	return sInstance->getLineShader_impl();
}

ShaderProgram *RendererShaders::getGizmoShader()
{
	return sInstance->getGizmoShader_impl();
}

ShaderProgram *RendererShaders::getGridShader()
{
	return sInstance->getGridShader_impl();
}

std::string RendererShaders::getStandardVertexShader()
{
	return sInstance->getStandardVertexShader_impl();
}

std::string RendererShaders::getStandardFragmentShader()
{
	return sInstance->getStandardFragmentShader_impl();
}