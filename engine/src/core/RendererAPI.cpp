#include "../../include/core/RendererAPI.h"
#include "../../include/core/PlatformDetection.h"

#define PHYSICSENGINE_RENDERER_API_OPENGL

#ifdef PHYSICSENGINE_PLATFORM_WIN32
#ifdef PHYSICSENGINE_RENDERER_API_OPENGL
#include "../../include/core/platform/RendererAPI_win32_opengl.h"
#elif PHYSICSENGINE_RENDERER_API_DIRECTX
#include "../../include/core/platform/RendererAPI_win32_directx.h"
#endif
#endif

using namespace PhysicsEngine;

RendererAPI::RendererAPI()
{

}

RendererAPI::~RendererAPI()
{

}

RendererAPI* RendererAPI::createRendererAPI()
{
#ifdef PHYSICSENGINE_PLATFORM_WIN32
#ifdef PHYSICSENGINE_RENDERER_API_OPENGL
	return new RendererAPI_win32_opengl();
#elif PHYSICSENGINE_RENDERER_API_DIRECTX
	return new RendererAPI_win32_directx();
#endif
#elif
	return nullptr;
#endif
}