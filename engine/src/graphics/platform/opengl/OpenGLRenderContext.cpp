#include "../../../../include/graphics/platform/opengl/OpenGLRenderContext.h"
#include "../../../../include/core/Log.h"

#include <GL/glew.h>

using namespace PhysicsEngine;

OpenGLRenderContext::OpenGLRenderContext(void* window)
{
    // Prepare OpenGlContext
    PIXELFORMATDESCRIPTOR desiredPixelFormat = {};
    desiredPixelFormat.nSize = sizeof(desiredPixelFormat);
    desiredPixelFormat.nVersion = 1;
    desiredPixelFormat.dwFlags = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER;
    desiredPixelFormat.cColorBits = 32;
    desiredPixelFormat.cAlphaBits = 8;
    desiredPixelFormat.iLayerType = PFD_MAIN_PLANE;

    mWindowDC = GetDC(static_cast<HWND>(window));

    int suggestedPixelFormatIndex = ChoosePixelFormat(mWindowDC, &desiredPixelFormat);

    PIXELFORMATDESCRIPTOR suggestedPixelFormat;
    DescribePixelFormat(mWindowDC, suggestedPixelFormatIndex, sizeof(suggestedPixelFormat),
        &suggestedPixelFormat);
    SetPixelFormat(mWindowDC, suggestedPixelFormatIndex, &suggestedPixelFormat);

    mOpenGLRC = wglCreateContext(mWindowDC);
    if (wglMakeCurrent(mWindowDC, mOpenGLRC))
    {
        SetSwapInterval(1);
    }
    else
    {
        // TODO: Handle error?
        return;
    }

    //ReleaseDC(static_cast<HWND>(window), mWindowDC);

    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        Log::error("OpenGL initialization failed");
    }
}

OpenGLRenderContext::~OpenGLRenderContext()
{
    wglDeleteContext(mOpenGLRC);
}

void OpenGLRenderContext::present()
{
    SwapBuffers(mWindowDC);
}

void OpenGLRenderContext::turnVsyncOn()
{
    SetSwapInterval(1);
}

void OpenGLRenderContext::turnVsyncOff()
{
    SetSwapInterval(0);
}

typedef BOOL(WINAPI* PFNWGLSWAPINTERVALEXTPROC)(int interval);
typedef const char* (WINAPI* PFNWGLGETEXTENSIONSSTRINGEXTPROC)(void);

// Got from https://stackoverflow.com/questions/589064/how-to-enable-vertical-sync-in-opengl/589232
static bool WGLExtensionSupported(const char* extension_name)
{
    // determine pointer to wglGetExtensionsStringEXT function
    PFNWGLGETEXTENSIONSSTRINGEXTPROC wglGetExtensionsStringEXT = (PFNWGLGETEXTENSIONSSTRINGEXTPROC)wglGetProcAddress("wglGetExtensionsStringEXT");

    if (wglGetExtensionsStringEXT)
    {
        if (strstr(wglGetExtensionsStringEXT(), extension_name) == NULL)
        {
            // string was not found
            return false;
        }
    }
    else
    {
        return false;
    }

    // extension is supported
    return true;
}

void OpenGLRenderContext::SetSwapInterval(int interval)
{
    if (WGLExtensionSupported("WGL_EXT_swap_control"))
    {
        PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
        if (wglSwapIntervalEXT)
        {
            wglSwapIntervalEXT(interval);
        }
    }
}

