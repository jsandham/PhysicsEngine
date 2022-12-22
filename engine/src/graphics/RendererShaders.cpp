#include "../../include/graphics/RendererShaders.h"
#include "../../include/graphics/RenderContext.h"
#include "../../include/core/Log.h"

#include "../../include/graphics/platform/opengl/OpenGLRendererShaders.h"
#include "../../include/graphics/platform/directx/DirectXRendererShaders.h"

using namespace PhysicsEngine;

StandardShader *RendererShaders::sStandardShader = nullptr;
SSAOShader* RendererShaders::sSSAOShader = nullptr;
GeometryShader* RendererShaders::sGeometryShader = nullptr;
DepthShader*  RendererShaders::sDepthShader = nullptr;
DepthCubemapShader* RendererShaders::sDepthCubemapShader = nullptr;
QuadShader* RendererShaders::sQuadShader = nullptr;
SpriteShader* RendererShaders::sSpriteShader = nullptr;
GBufferShader* RendererShaders::sGBufferShader = nullptr;
ColorShader* RendererShaders::sColorShader = nullptr;
ColorInstancedShader* RendererShaders::sColorInstancedShader = nullptr;
NormalShader* RendererShaders::sNormalShader = nullptr;
NormalInstancedShader* RendererShaders::sNormalInstancedShader = nullptr;
PositionShader* RendererShaders::sPositionShader = nullptr;
PositionInstancedShader* RendererShaders::sPositionInstancedShader = nullptr;
LinearDepthShader* RendererShaders::sLinearDepthShader = nullptr;
LinearDepthInstancedShader* RendererShaders::sLinearDepthInstancedShader = nullptr;
LineShader* RendererShaders::sLineShader = nullptr;
GizmoShader* RendererShaders::sGizmoShader = nullptr;
GridShader* RendererShaders::sGridShader = nullptr;

void RendererShaders::createInternalShaders()
{
    Log::warn("Start compling internal shaders\n");
    sStandardShader = StandardShader::create();
    sSSAOShader = SSAOShader::create();
    sGeometryShader = GeometryShader::create();
    sDepthShader = DepthShader::create();
    sDepthCubemapShader = DepthCubemapShader::create();
    sQuadShader = QuadShader::create();
    sSpriteShader = SpriteShader::create();
    sGBufferShader = GBufferShader::create();
    sColorShader = ColorShader::create();
    sColorInstancedShader = ColorInstancedShader::create();
    sNormalShader = NormalShader::create();
    sNormalInstancedShader = NormalInstancedShader::create();
    sPositionShader = PositionShader::create();
    sPositionInstancedShader = PositionInstancedShader::create();
    sLinearDepthShader = LinearDepthShader::create();
    sLinearDepthInstancedShader = LinearDepthInstancedShader::create();
    sLineShader = LineShader::create();
    sGizmoShader = GizmoShader::create();
    sGridShader = GridShader::create();
    Log::warn("Finished compiling internal shaders\n");
}

StandardShader *RendererShaders::getStandardShader()
{
    return RendererShaders::sStandardShader;
}

SSAOShader *RendererShaders::getSSAOShader()
{
    return RendererShaders::sSSAOShader;
}

GeometryShader *RendererShaders::getGeometryShader()
{
    return RendererShaders::sGeometryShader;
}

DepthShader *RendererShaders::getDepthShader()
{
    return RendererShaders::sDepthShader;
}

DepthCubemapShader *RendererShaders::getDepthCubemapShader()
{
    return RendererShaders::sDepthCubemapShader;
}

QuadShader *RendererShaders::getScreenQuadShader()
{
    return RendererShaders::sQuadShader;
}

SpriteShader *RendererShaders::getSpriteShader()
{
    return RendererShaders::sSpriteShader;
}

GBufferShader *RendererShaders::getGBufferShader()
{
    return RendererShaders::sGBufferShader;
}

ColorShader *RendererShaders::getColorShader()
{
    return RendererShaders::sColorShader;
}

ColorInstancedShader *RendererShaders::getColorInstancedShader()
{
    return RendererShaders::sColorInstancedShader;
}

NormalShader *RendererShaders::getNormalShader()
{
    return RendererShaders::sNormalShader;
}

NormalInstancedShader *RendererShaders::getNormalInstancedShader()
{
    return RendererShaders::sNormalInstancedShader;
}

PositionShader *RendererShaders::getPositionShader()
{
    return RendererShaders::sPositionShader;
}

PositionInstancedShader *RendererShaders::getPositionInstancedShader()
{
    return RendererShaders::sPositionInstancedShader;
}

LinearDepthShader *RendererShaders::getLinearDepthShader()
{
    return RendererShaders::sLinearDepthShader;
}

LinearDepthInstancedShader *RendererShaders::getLinearDepthInstancedShader()
{
    return RendererShaders::sLinearDepthInstancedShader;
}

LineShader *RendererShaders::getLineShader()
{
    return RendererShaders::sLineShader;
}

GizmoShader *RendererShaders::getGizmoShader()
{
    return RendererShaders::sGizmoShader;
}

GridShader *RendererShaders::getGridShader()
{
    return RendererShaders::sGridShader;
}

StandardShader *StandardShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLStandardShader();
    case RenderAPI::DirectX:
        return new DirectXStandardShader();
    }

    return nullptr;
}

GBufferShader *GBufferShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLGBufferShader();
    case RenderAPI::DirectX:
        return new DirectXGBufferShader();
    }

    return nullptr;
}

QuadShader *QuadShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLQuadShader();
    case RenderAPI::DirectX:
        return new DirectXQuadShader();
    }

    return nullptr;
}

DepthShader *DepthShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLDepthShader();
    case RenderAPI::DirectX:
        return new DirectXDepthShader();
    }

	return nullptr;
}

DepthCubemapShader *DepthCubemapShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLDepthCubemapShader();
    case RenderAPI::DirectX:
        return new DirectXDepthCubemapShader();
    }

    return nullptr;
}

GeometryShader *GeometryShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLGeometryShader();
    case RenderAPI::DirectX:
        return new DirectXGeometryShader();
    }

    return nullptr;
}

NormalShader *NormalShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLNormalShader();
    case RenderAPI::DirectX:
        return new DirectXNormalShader();
    }

    return nullptr;
}

NormalInstancedShader *NormalInstancedShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLNormalInstancedShader();
    case RenderAPI::DirectX:
        return new DirectXNormalInstancedShader();
    }

    return nullptr;
}

PositionShader *PositionShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLPositionShader();
    case RenderAPI::DirectX:
        return new DirectXPositionShader();
    }

    return nullptr;
}

PositionInstancedShader *PositionInstancedShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLPositionInstancedShader();
    case RenderAPI::DirectX:
        return new DirectXPositionInstancedShader();
    }

    return nullptr;
}

LinearDepthShader *LinearDepthShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLLinearDepthShader();
    case RenderAPI::DirectX:
        return new DirectXLinearDepthShader();
    }

    return nullptr;
}

LinearDepthInstancedShader *LinearDepthInstancedShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLLinearDepthInstancedShader();
    case RenderAPI::DirectX:
        return new DirectXLinearDepthInstancedShader();
    }

    return nullptr;
}

ColorShader *ColorShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLColorShader();
    case RenderAPI::DirectX:
        return new DirectXColorShader();
    }

    return nullptr;
}

ColorInstancedShader *ColorInstancedShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLColorInstancedShader();
    case RenderAPI::DirectX:
        return new DirectXColorInstancedShader();
    }

    return nullptr;
}

SSAOShader *SSAOShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLSSAOShader();
    case RenderAPI::DirectX:
        return new DirectXSSAOShader();
    }

    return nullptr;
}

SpriteShader *SpriteShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLSpriteShader();
    case RenderAPI::DirectX:
        return new DirectXSpriteShader();
    }

    return nullptr;
}

LineShader *LineShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLLineShader();
    case RenderAPI::DirectX:
        return new DirectXLineShader();
    }

    return nullptr;
}

GizmoShader *GizmoShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLGizmoShader();
    case RenderAPI::DirectX:
        return new DirectXGizmoShader();
    }

    return nullptr;
}

GridShader *GridShader::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLGridShader();
    case RenderAPI::DirectX:
        return new DirectXGridShader();
    }

    return nullptr;
}