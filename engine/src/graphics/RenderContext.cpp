#include "../../include/core/PlatformDetection.h"
#include "../../include/graphics/RenderContext.h"

#ifdef PHYSICSENGINE_PLATFORM_WIN32
#include "../../include/graphics/platform/opengl/OpenGLRenderContext.h"
#include "../../include/graphics/platform/directx/DirectXRenderContext.h"
#endif

using namespace PhysicsEngine;

RenderAPI RenderContext::sAPI = RenderAPI::OpenGL; // RenderAPI::DirectX;
RenderContext* RenderContext::sContext = nullptr;

RenderAPI RenderContext::getRenderAPI()
{
    return sAPI;
}

void RenderContext::setRenderAPI(RenderAPI api)
{
    sAPI = api;
}

void RenderContext::createRenderContext(void* window)
{
#ifdef PHYSICSENGINE_PLATFORM_WIN32
    switch (getRenderAPI())
    {
    case RenderAPI::OpenGL:
        sContext = new OpenGLRenderContext(window);
        break;
    case RenderAPI::DirectX:
        sContext = new DirectXRenderContext(window);
        break;
    default:
        sContext = nullptr;
    }
#elif
    sContext = nullptr;
#endif
}