#include "../../../include/core/platform/OpenGLApplicationGraphicsContext.h"

using namespace PhysicsEngine;

OpenGLApplicationGraphicsContext::OpenGLApplicationGraphicsContext(HWND window)
{
    // Prepare OpenGlContext
    CreateGlContext(window);
    SetSwapInterval(1);
    glewInit();
}

OpenGLApplicationGraphicsContext::~OpenGLApplicationGraphicsContext()
{
    wglDeleteContext(g_GLRenderContext);
}

void OpenGLApplicationGraphicsContext::update()
{
    SwapBuffers(g_HDCDeviceContext);
}

void OpenGLApplicationGraphicsContext::CreateGlContext(HWND window)
{
    PIXELFORMATDESCRIPTOR desiredPixelFormat = {};
    desiredPixelFormat.nSize = sizeof(desiredPixelFormat);
    desiredPixelFormat.nVersion = 1;
    desiredPixelFormat.dwFlags = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER; // PFD_DRAW_TO_BITMAP
    desiredPixelFormat.cColorBits = 32;
    desiredPixelFormat.cAlphaBits = 8;
    desiredPixelFormat.iLayerType = PFD_MAIN_PLANE;

    g_HDCDeviceContext = GetDC(window);

    int suggestedPixelFormatIndex = ChoosePixelFormat(g_HDCDeviceContext, &desiredPixelFormat);

    PIXELFORMATDESCRIPTOR suggestedPixelFormat;
    DescribePixelFormat(g_HDCDeviceContext, suggestedPixelFormatIndex, sizeof(suggestedPixelFormat),
        &suggestedPixelFormat);
    SetPixelFormat(g_HDCDeviceContext, suggestedPixelFormatIndex, &suggestedPixelFormat);

    g_GLRenderContext = wglCreateContext(g_HDCDeviceContext);
    if (!wglMakeCurrent(g_HDCDeviceContext, g_GLRenderContext))
    {
        return;
    }

    // PIXELFORMATDESCRIPTOR pfd =
    //{
    //	sizeof(PIXELFORMATDESCRIPTOR),
    //	1,
    //	PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,    //Flags
    //	PFD_TYPE_RGBA,        // The kind of framebuffer. RGBA or palette.
    //	32,                   // Colordepth of the framebuffer.
    //	0, 0, 0, 0, 0, 0,
    //	0,
    //	0,
    //	0,
    //	0, 0, 0, 0,
    //	24,                   // Number of bits for the depthbuffer
    //	8,                    // Number of bits for the stencilbuffer
    //	0,                    // Number of Aux buffers in the framebuffer.
    //	PFD_MAIN_PLANE,
    //	0,
    //	0, 0, 0
    //};

    // g_HDCDeviceContext = GetDC(g_hwnd);

    // int pixelFormal = ChoosePixelFormat(g_HDCDeviceContext, &pfd);
    // SetPixelFormat(g_HDCDeviceContext, pixelFormal, &pfd);
    // g_GLRenderContext = wglCreateContext(g_HDCDeviceContext);
    // wglMakeCurrent(g_HDCDeviceContext, g_GLRenderContext);
}

bool OpenGLApplicationGraphicsContext::SetSwapInterval(int interval)
{
    if (WGLExtensionSupported("WGL_EXT_swap_control"))
    {
        // Extension is supported, init pointers.
        wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");

        // this is another function from WGL_EXT_swap_control extension
        wglGetSwapIntervalEXT = (PFNWGLGETSWAPINTERVALEXTPROC)wglGetProcAddress("wglGetSwapIntervalEXT");

        wglSwapIntervalEXT(interval);
        return true;
    }

    return false;
}

// Got from https://stackoverflow.com/questions/589064/how-to-enable-vertical-sync-in-opengl/589232
bool OpenGLApplicationGraphicsContext::WGLExtensionSupported(const char* extension_name)
{
    // this is pointer to function which returns pointer to string with list of all wgl extensions
    PFNWGLGETEXTENSIONSSTRINGEXTPROC _wglGetExtensionsStringEXT = NULL;

    // determine pointer to wglGetExtensionsStringEXT function
    _wglGetExtensionsStringEXT = (PFNWGLGETEXTENSIONSSTRINGEXTPROC)wglGetProcAddress("wglGetExtensionsStringEXT");

    if (strstr(_wglGetExtensionsStringEXT(), extension_name) == NULL)
    {
        // string was not found
        return false;
    }

    // extension is supported
    return true;
}