#pragma warning(disable : 4996)
#define _CRTDBG_MAP_ALLOC

#define GLM_FORCE_RADIANS

#include <GL/glew.h>
#include <gl/gl.h>

#include <crtdbg.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <tchar.h>
#include <thread>
#include <windows.h>
#include <windowsx.h>
#include <xinput.h>

#include <core/WorldManager.h>
#include <core/Time.h>
#include <core/Input.h>

using namespace PhysicsEngine;

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

static bool application_quit = false;

// =============================================================================
//                                  DEFINES/MACROS
// =============================================================================

// =============================================================================
//                               GLOBAL VARIABLES
// =============================================================================
HGLRC g_GLRenderContext;
HDC g_HDCDeviceContext;
HWND g_hwnd;
PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT;
PFNWGLGETSWAPINTERVALEXTPROC wglGetSwapIntervalEXT;
int g_display_w = 1920;
int g_display_h = 1080;

// =============================================================================
//                             FOWARD DECLARATIONS
// =============================================================================
void CreateGlContext();
void SetCurrentContext();
bool SetSwapInterval(int interval); // 0 - No Interval, 1 - Sync whit VSYNC, n - n times Sync with VSYNC
bool WGLExtensionSupported(const char* extension_name);
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

void RefreshDirectory(LPTSTR);
void RefreshTree(LPTSTR);
void WatchDirectory(LPTSTR);

// =============================================================================
//                            CORE MAIN FUNCTIONS
// =============================================================================
//
// Application Entry
//------------------------------------------------------------------------------
int WINAPI wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine,
    _In_ int nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);
    UNREFERENCED_PARAMETER(nCmdShow);

    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hbrBackground = (HBRUSH)(COLOR_BACKGROUND);
    wc.lpszClassName = _T("NCUI");
    wc.style = CS_OWNDC;
    if (!RegisterClass(&wc))
        return 1;
    g_hwnd = CreateWindowEx(0, wc.lpszClassName, _T("Game"), WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT,
        CW_USEDEFAULT, g_display_w, g_display_h, 0, 0, hInstance, 0);

    // Show the window
    ShowWindow(g_hwnd, SW_SHOWDEFAULT);
    UpdateWindow(g_hwnd);

    // Prepare OpenGlContext
    CreateGlContext();
    SetSwapInterval(1);
    glewInit();



    AllocConsole();
    freopen("CONIN$", "r", stdin);
    freopen("CONOUT$", "w", stdout);
    freopen("CONOUT$", "w", stderr);

    WorldManager manager;

    manager.init();

    HWND prevActiveWindow = NULL;
    HWND activeWindow = NULL;

    // Main loop
    MSG msg;
    ZeroMemory(&msg, sizeof(msg));
    while (msg.message != WM_QUIT)
    {
        prevActiveWindow = activeWindow;
        activeWindow = GetActiveWindow();

        // Poll and handle messages (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your
        // inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those
        // two flags.
        while (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE) != 0)
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        manager.update(Time(), Input());

        if (application_quit)
        {
            break;
        }

        wglMakeCurrent(g_HDCDeviceContext, g_GLRenderContext);

        SwapBuffers(g_HDCDeviceContext);
    }

    wglDeleteContext(g_GLRenderContext);

    DestroyWindow(g_hwnd);
    UnregisterClass(_T("NCUI"), wc.hInstance);

    return 0;
}

//extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    //if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
    //    return 0;

    switch (msg)
    {
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
            return 0;
        break;
    case WM_DESTROY:
        application_quit = true;
        PostQuitMessage(0);
        return 0;
    }

    return DefWindowProc(hWnd, msg, wParam, lParam);
}

void CreateGlContext()
{
    PIXELFORMATDESCRIPTOR desiredPixelFormat = {};
    desiredPixelFormat.nSize = sizeof(desiredPixelFormat);
    desiredPixelFormat.nVersion = 1;
    desiredPixelFormat.dwFlags = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER; // PFD_DRAW_TO_BITMAP
    desiredPixelFormat.cColorBits = 32;
    desiredPixelFormat.cAlphaBits = 8;
    desiredPixelFormat.iLayerType = PFD_MAIN_PLANE;

    g_HDCDeviceContext = GetDC(g_hwnd);

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
}

bool SetSwapInterval(int interval)
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
bool WGLExtensionSupported(const char* extension_name)
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