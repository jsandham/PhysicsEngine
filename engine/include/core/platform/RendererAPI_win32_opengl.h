#ifndef RENDERER_API_WIN32_OPENGL_H__
#define RENDERER_API_WIN32_OPENGL_H__

#include "../RendererAPI.h"

#include <windows.h>

namespace PhysicsEngine
{
	class RendererAPI_win32_opengl : public RendererAPI
	{
	private:
		HGLRC mOpenGLRC;
		HDC mWindowDC;

	public:
		RendererAPI_win32_opengl();
		~RendererAPI_win32_opengl();

		void init(void* window) override;
		void update() override;
		void cleanup() override;
        void turnVsyncOn() override;
        void turnVsyncOff() override;

	private:
		void CreateGlContext(void* window);
		void SetSwapInterval(int interval);
	};
}

#endif RENDERER_API_WIN32_OPENGL_H__