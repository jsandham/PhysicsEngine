#ifndef RENDER_CONTEXT_OPENGL_H__
#define RENDER_CONTEXT_OPENGL_H__

#include "../../RenderContext.h"

#include <windows.h>

namespace PhysicsEngine
{
	class OpenGLRenderContext : public RenderContext
	{
	private:
		HGLRC mOpenGLRC;
		HDC mWindowDC;

	public:
		OpenGLRenderContext(void* window);
		~OpenGLRenderContext();

		void present();
		void turnVsyncOn();
		void turnVsyncOff();

		static OpenGLRenderContext* get() { return (OpenGLRenderContext*)sContext; }

	private:
		static void SetSwapInterval(int interval);
	};
}

#endif // RENDER_CONTEXT_OPENGL_H__