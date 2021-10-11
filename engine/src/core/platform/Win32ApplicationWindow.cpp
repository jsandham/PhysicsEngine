#include "../../../include/core/platform/Win32ApplicationWindow.h"

#include <tchar.h>

using namespace PhysicsEngine;

extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return 0;

    switch (msg)
    {
        /*case WM_SIZE:
            if (wParam != SIZE_MINIMIZED)
            {
                g_display_w = (UINT)LOWORD(lParam);
                g_display_h = (UINT)HIWORD(lParam);
            }
            return 0;*/
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
            return 0;
        break;
    case WM_DESTROY:
        //application_quit = true;
        PostQuitMessage(0);
        return 0;
    }

    return DefWindowProc(hWnd, msg, wParam, lParam);
}

Win32ApplicationWindow::Win32ApplicationWindow(const std::string& title, int width, int height) : ApplicationWindow()
{
	init(title, width, height);

    mRendererAPI = RendererAPI::createRendererAPI();

    mRendererAPI->init(static_cast<void*>(mWindow));
}

Win32ApplicationWindow::~Win32ApplicationWindow()
{
	cleanup();

    mRendererAPI->cleanup();
    delete mRendererAPI;
}

void Win32ApplicationWindow::update()
{
    // Poll and handle messages (inputs, window resize, etc.)
    MSG message;
    while (PeekMessage(&message, NULL, 0U, 0U, PM_REMOVE) != 0)
    {
        TranslateMessage(&message);
        DispatchMessage(&message);
    }

    mRendererAPI->update();
}

int Win32ApplicationWindow::getWidth() const
{
	return mWidth;
}

int Win32ApplicationWindow::getHeight() const
{
	return mHeight;
}

void* Win32ApplicationWindow::getNativeWindow() const
{
    return static_cast<void*>(mWindow);
}

void Win32ApplicationWindow::init(const std::string& title, int width, int height)
{
    mTitle = title;
    mWidth = width;
    mHeight = height;

    mWC = { 0 };
    mWC.lpfnWndProc = WndProc;
    mWC.hInstance = GetModuleHandle(0);
    mWC.hbrBackground = (HBRUSH)(COLOR_BACKGROUND);
    mWC.lpszClassName = _T("NCUI");
    mWC.style = CS_OWNDC;
    if (!RegisterClass(&mWC))
        return;
    mWindow = CreateWindowEx(0, mWC.lpszClassName, _T(mTitle.c_str()), WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT,
        CW_USEDEFAULT, mWidth, mHeight, 0, 0, mWC.hInstance, 0);

    // Show the window
    ShowWindow(mWindow, SW_SHOWDEFAULT);
    UpdateWindow(mWindow);
}

void Win32ApplicationWindow::cleanup()
{
    DestroyWindow(mWindow);
    UnregisterClass(_T("NCUI"), mWC.hInstance);
}