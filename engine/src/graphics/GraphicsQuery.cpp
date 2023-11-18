#include "../../include/graphics/GraphicsQuery.h"
#include "../../include/graphics/RenderContext.h"

#include "../../include/graphics/platform/directx/DirectXGraphicsQuery.h"
#include "../../include/graphics/platform/opengl/OpenGLGraphicsQuery.h"

using namespace PhysicsEngine;

OcclusionQuery::OcclusionQuery()
{
}

OcclusionQuery::~OcclusionQuery()
{
}

OcclusionQuery *OcclusionQuery::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLOcclusionQuery();
    case RenderAPI::DirectX:
        return new DirectXOcclusionQuery();
    }

    return nullptr;
}