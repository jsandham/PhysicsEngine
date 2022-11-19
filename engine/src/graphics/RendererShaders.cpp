#include "../../include/graphics/RendererShaders.h"
#include "../../include/graphics/RenderContext.h"

#include "../../include/graphics/platform/opengl/OpenGLRendererShaders.h"
#include "../../include/graphics/platform/directx/DirectXRendererShaders.h"

using namespace PhysicsEngine;

RendererShaders* RendererShaders::sInstance = nullptr;

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

	sInstance->init_impl();
}

RendererShaders* RendererShaders::getRendererShaders()
{
	return sInstance;
}

SSAOShader RendererShaders::getSSAOShader()
{
	return sInstance->getSSAOShader_impl();
}

GeometryShader RendererShaders::getGeometryShader()
{
	return sInstance->getGeometryShader_impl();
}

DepthShader RendererShaders::getDepthShader()
{
	return sInstance->getDepthShader_impl();
}

DepthCubemapShader RendererShaders::getDepthCubemapShader()
{
	return sInstance->getDepthCubemapShader_impl();
}

ScreenQuadShader RendererShaders::getScreenQuadShader()
{
	return sInstance->getScreenQuadShader_impl();
}

SpriteShader RendererShaders::getSpriteShader()
{
	return sInstance->getSpriteShader_impl();
}

GBufferShader RendererShaders::getGBufferShader()
{
	return sInstance->getGBufferShader_impl();
}

ColorShader RendererShaders::getColorShader()
{
	return sInstance->getColorShader_impl();
}

ColorInstancedShader RendererShaders::getColorInstancedShader()
{
	return sInstance->getColorInstancedShader_impl();
}

NormalShader RendererShaders::getNormalShader()
{
	return sInstance->getNormalShader_impl();
}

NormalInstancedShader RendererShaders::getNormalInstancedShader()
{
	return sInstance->getNormalInstancedShader_impl();
}

PositionShader RendererShaders::getPositionShader()
{
	return sInstance->getPositionShader_impl();
}

PositionInstancedShader RendererShaders::getPositionInstancedShader()
{
	return sInstance->getPositionInstancedShader_impl();
}

LinearDepthShader RendererShaders::getLinearDepthShader()
{
	return sInstance->getLinearDepthShader_impl();
}

LinearDepthInstancedShader RendererShaders::getLinearDepthInstancedShader()
{
	return sInstance->getLinearDepthInstancedShader_impl();
}

LineShader RendererShaders::getLineShader()
{
	return sInstance->getLineShader_impl();
}

GizmoShader RendererShaders::getGizmoShader()
{
	return sInstance->getGizmoShader_impl();
}

GridShader RendererShaders::getGridShader()
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