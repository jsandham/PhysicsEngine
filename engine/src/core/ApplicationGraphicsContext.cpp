#include "../../include/core/ApplicationGraphicsContext.h"
#include "../../include/core/platform/OpenGLApplicationGraphicsContext.h"

using namespace PhysicsEngine;

ApplicationGraphicsContext::ApplicationGraphicsContext()
{

}

ApplicationGraphicsContext::~ApplicationGraphicsContext()
{

}

ApplicationGraphicsContext* ApplicationGraphicsContext::createApplicationGraphicsContext(void* window)
{
	//#ifdef PHYSICSENGINE_GRAPHICS_OPENGL
	return new OpenGLApplicationGraphicsContext(static_cast<HWND>(window));
	//#endif

		//return nullptr;
}