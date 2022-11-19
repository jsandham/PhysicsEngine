#include "../../include/core/PlatformDetection.h"
#include "../../include/graphics/RenderContext.h"

#ifdef PHYSICSENGINE_PLATFORM_WIN32
#include "../../include/graphics/platform/opengl/RenderContextOpenGL.h"
#include "../../include/graphics/platform/directx/RenderContextDirectX.h"
#endif

using namespace PhysicsEngine;

RenderAPI RenderContext::sAPI = RenderAPI::OpenGL;
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
        sContext = new RenderContextOpenGL(window);
        break;
    case RenderAPI::DirectX:
        sContext = new RenderContextDirectX(window);
        break;
    default:
        sContext = nullptr;
    }
#elif
    sContext = nullptr;
#endif
}


//#include "../../include/core/RenderContext.h"
//#include "../../include/core/PlatformDetection.h"
//
//#ifdef PHYSICSENGINE_PLATFORM_WIN32
//#include "../../include/core/platform/RenderContext_win32_opengl.h"
//#include "../../include/core/platform/RenderContext_win32_directx.h"
//#endif
//
//using namespace PhysicsEngine;
//
//RenderContext::RenderContext()
//{
//
//}
//
//RenderContext::~RenderContext()
//{
//
//}
//
//RenderAPI RenderContext::sAPI = RenderAPI::DirectX;// OpenGL;
//
//RenderAPI RenderContext::getRenderAPI()
//{
//    return sAPI;
//}
//
//void RenderContext::setRenderAPI(RenderAPI api)
//{
//    sAPI = api;
//}
//
//RenderContext *RenderContext::createRenderContext()
//{
//#ifdef PHYSICSENGINE_PLATFORM_WIN32
//    switch (getRenderAPI())
//    {
//    case RenderAPI::OpenGL:
//        return new RenderContext_win32_opengl();
//    case RenderAPI::DirectX:
//        return new RenderContext_win32_directx();
//    default:
//        return nullptr;
//	}
//#elif
//	return nullptr;
//#endif
//}