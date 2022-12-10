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


//#ifndef RENDER_CONTEXT_WIN32_OPENGL_H__
//#define RENDER_CONTEXT_WIN32_OPENGL_H__
//
//#include "../RenderContext.h"
//
//#include <windows.h>
//
//namespace PhysicsEngine
//{
//	class RenderContext_win32_opengl : public RenderContext
//	{
//	private:
//		HGLRC mOpenGLRC;
//		HDC mWindowDC;
//
//	public:
//        RenderContext_win32_opengl();
//      ~RenderContext_win32_opengl();
//
//		void init(void* window) override;
//		void update() override;
//		void cleanup() override;
//        void turnVsyncOn() override;
//        void turnVsyncOff() override;
//
//	private:
//		void CreateGlContext(void* window);
//		void SetSwapInterval(int interval);
//	};
//}
//
//#endif // RENDER_CONTEXT_WIN32_OPENGL_H__