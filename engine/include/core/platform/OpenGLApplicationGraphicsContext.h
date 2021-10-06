#ifndef OPENGL_APPLICATION_GRAPHICS_CONTEXT_H__
#define OPENGL_APPLICATION_GRAPHICS_CONTEXT_H__

#include <windows.h>

#include "../ApplicationGraphicsContext.h"

#include "../../../../external/glew-2.1.0/GL/glew.h"

#include <gl/gl.h>

namespace PhysicsEngine
{
#ifndef WGL_EXT_extensions_string
#define WGL_EXT_extensions_string 1
#ifdef WGL_WGLEXT_PROTOTYPES
	extern const char* WINAPI wglGetExtensionsStringEXT(void);
#endif /* WGL_WGLEXT_PROTOTYPES */
	typedef const char* (WINAPI* PFNWGLGETEXTENSIONSSTRINGEXTPROC)(void);
#endif

#ifndef WGL_EXT_swap_control
#define WGL_EXT_swap_control 1
#ifdef WGL_WGLEXT_PROTOTYPES
	extern BOOL WINAPI wglSwapIntervalEXT(int);
	extern int WINAPI wglGetSwapIntervalEXT(void);
#endif /* WGL_WGLEXT_PROTOTYPES */
	typedef BOOL(WINAPI* PFNWGLSWAPINTERVALEXTPROC)(int interval);
	typedef int(WINAPI* PFNWGLGETSWAPINTERVALEXTPROC)(void);
#endif

	class OpenGLApplicationGraphicsContext : public ApplicationGraphicsContext
	{
	private:
		HGLRC g_GLRenderContext;
		HDC g_HDCDeviceContext;

		PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT;
		PFNWGLGETSWAPINTERVALEXTPROC wglGetSwapIntervalEXT;

	public:
		OpenGLApplicationGraphicsContext(HWND window);
		~OpenGLApplicationGraphicsContext();

		void update() override;

	private:
		void CreateGlContext(HWND window);
		bool SetSwapInterval(int interval);
		bool WGLExtensionSupported(const char* extension_name);
	};
}

#endif OPENGL_APPLICATION_GRAPHICS_CONTEXT_H__