#include "../../../../include/graphics/platform/opengl/RenderContextOpenGL.h"
#include "../../../../include/core/Log.h"

#include <GL/glew.h>

using namespace PhysicsEngine;

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

RenderContextOpenGL::RenderContextOpenGL(void* window)
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

RenderContextOpenGL::~RenderContextOpenGL()
{
    wglDeleteContext(mOpenGLRC);
}

void RenderContextOpenGL::present()
{
    SwapBuffers(mWindowDC);
}

void RenderContextOpenGL::turnVsyncOn()
{
    SetSwapInterval(1);
}

void RenderContextOpenGL::turnVsyncOff()
{
    SetSwapInterval(0);
}

void RenderContextOpenGL::SetSwapInterval(int interval)
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





//#include "../../../include/core/platform/RenderContext_win32_opengl.h"
//#include "../../../include/core/Log.h"
//
//#include <GL/glew.h>
//
//using namespace PhysicsEngine;
//
//typedef BOOL (WINAPI* PFNWGLSWAPINTERVALEXTPROC)(int interval);
//typedef const char* (WINAPI* PFNWGLGETEXTENSIONSSTRINGEXTPROC)(void);
//
//// Got from https://stackoverflow.com/questions/589064/how-to-enable-vertical-sync-in-opengl/589232
//static bool WGLExtensionSupported(const char* extension_name)
//{
//    // determine pointer to wglGetExtensionsStringEXT function
//    PFNWGLGETEXTENSIONSSTRINGEXTPROC wglGetExtensionsStringEXT = (PFNWGLGETEXTENSIONSSTRINGEXTPROC)wglGetProcAddress("wglGetExtensionsStringEXT");
//
//    if (wglGetExtensionsStringEXT)
//    {
//        if (strstr(wglGetExtensionsStringEXT(), extension_name) == NULL)
//        {
//            // string was not found
//            return false;
//        }
//    }
//    else
//    {
//        return false;
//    }
//
//    // extension is supported
//    return true;
//}
//
//RenderContext_win32_opengl::RenderContext_win32_opengl()
//{    
//}
//
//RenderContext_win32_opengl::~RenderContext_win32_opengl()
//{
//}
//
//void RenderContext_win32_opengl::init(void *window)
//{
//    // Prepare OpenGlContext
//    CreateGlContext(window);
//
//    GLenum err = glewInit();
//    if (GLEW_OK != err)
//    {
//        Log::error("OpenGL initialization failed");
//    }
//}
//
//void RenderContext_win32_opengl::update()
//{
//    SwapBuffers(mWindowDC);
//}
//
//void RenderContext_win32_opengl::cleanup()
//{
//    wglDeleteContext(mOpenGLRC);
//}
//
//void RenderContext_win32_opengl::turnVsyncOn()
//{
//    SetSwapInterval(1);
//}
//
//void RenderContext_win32_opengl::turnVsyncOff()
//{
//    SetSwapInterval(0);
//}
//
//void RenderContext_win32_opengl::CreateGlContext(void *window)
//{
//    PIXELFORMATDESCRIPTOR desiredPixelFormat = {};
//    desiredPixelFormat.nSize = sizeof(desiredPixelFormat);
//    desiredPixelFormat.nVersion = 1;
//    desiredPixelFormat.dwFlags = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER;
//    desiredPixelFormat.cColorBits = 32;
//    desiredPixelFormat.cAlphaBits = 8;
//    desiredPixelFormat.iLayerType = PFD_MAIN_PLANE;
//
//    mWindowDC = GetDC(static_cast<HWND>(window));
//
//    int suggestedPixelFormatIndex = ChoosePixelFormat(mWindowDC, &desiredPixelFormat);
//
//    PIXELFORMATDESCRIPTOR suggestedPixelFormat;
//    DescribePixelFormat(mWindowDC, suggestedPixelFormatIndex, sizeof(suggestedPixelFormat),
//        &suggestedPixelFormat);
//    SetPixelFormat(mWindowDC, suggestedPixelFormatIndex, &suggestedPixelFormat);
//
//    mOpenGLRC = wglCreateContext(mWindowDC);
//    if (wglMakeCurrent(mWindowDC, mOpenGLRC))
//    {
//        SetSwapInterval(1);
//    }
//    else
//    {
//        // TODO: Handle error?
//        return;
//    }
//
//    //ReleaseDC(static_cast<HWND>(window), mWindowDC);
//}
//
//void RenderContext_win32_opengl::SetSwapInterval(int interval)
//{
//    if (WGLExtensionSupported("WGL_EXT_swap_control"))
//    {
//        PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
//        if (wglSwapIntervalEXT)
//        {
//            wglSwapIntervalEXT(interval);
//        }
//    }
//}
