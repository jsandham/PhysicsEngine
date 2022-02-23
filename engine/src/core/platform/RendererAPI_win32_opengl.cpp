#include "../../../include/core/platform/RendererAPI_win32_opengl.h"
#include "../../../include/core/Log.h"

#include <GL/glew.h>

using namespace PhysicsEngine;

typedef BOOL (WINAPI* PFNWGLSWAPINTERVALEXTPROC)(int interval);
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

RendererAPI_win32_opengl::RendererAPI_win32_opengl()
{    
}

RendererAPI_win32_opengl::~RendererAPI_win32_opengl()
{
}

void RendererAPI_win32_opengl::init(void* window)
{
    // Prepare OpenGlContext
    CreateGlContext(window);

    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        Log::error("OpenGL initialization failed");
    }
}

void RendererAPI_win32_opengl::update()
{
    SwapBuffers(mWindowDC);
}

void RendererAPI_win32_opengl::cleanup()
{
    wglDeleteContext(mOpenGLRC);
}

void RendererAPI_win32_opengl::CreateGlContext(void* window)
{
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
        SetSwapInterval(0);
    }
    else
    {
        // TODO: Handle error?
        return;
    }

    //ReleaseDC(static_cast<HWND>(window), mWindowDC);
}

void RendererAPI_win32_opengl::SetSwapInterval(int interval)
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
