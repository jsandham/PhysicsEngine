#ifndef RENDERER_API_WIN32_OPENGL_H__
#define RENDERER_API_WIN32_OPENGL_H__

#include "../RendererAPI.h"

#include <windows.h>

namespace PhysicsEngine
{
	class RendererAPI_win32_opengl : public RendererAPI
	{
	private:
		HGLRC g_GLRenderContext;
		HDC g_HDCDeviceContext;

	public:
		RendererAPI_win32_opengl();
		~RendererAPI_win32_opengl();

		void init(void* window) override;
		void update() override;
		void cleanup() override;

	private:
		void CreateGlContext(void* window);
		bool SetSwapInterval(int interval);
		bool WGLExtensionSupported(const char* extension_name);
	};
}

#endif RENDERER_API_WIN32_OPENGL_H__