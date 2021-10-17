#include "../../../include/core/platform/Win32ApplicationWindow.h"

#include <tchar.h>

using namespace PhysicsEngine;

static bool mIsMinimized = false;

// define in application
extern LRESULT PhysicsEngine_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (PhysicsEngine_WndProcHandler(hWnd, msg, wParam, lParam))
        return 0;

    switch (msg)
    {
    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED)
        {
            mIsMinimized = true;
            return 0;
        }
        else if (wParam == SIZE_MAXIMIZED)
        {
            mIsMinimized = false;
            return 0;
        }
        break;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
            return 0;
        break;
    case WM_DESTROY:
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
        if (message.message == WM_QUIT)
        {
            mRunning = false;
        }

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

bool Win32ApplicationWindow::isRunning() const
{
    return mRunning;
}

bool Win32ApplicationWindow::isMinimized() const
{
    return mIsMinimized;
}

void Win32ApplicationWindow::init(const std::string& title, int width, int height)
{
    mTitle = title;
    mWidth = width;
    mHeight = height;

    mRunning = true;

    mWC = { 0 };
    mWC.lpfnWndProc = WndProc;
    mWC.hInstance = GetModuleHandle(0);
    mWC.hbrBackground = (HBRUSH)(COLOR_BACKGROUND);
    mWC.lpszClassName = _T("PHYSICS_ENGINE_WINDOW_CLASS");
    mWC.style = CS_OWNDC;// CS_HREDRAW | CS_VREDRAW;
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
    UnregisterClass(_T("PHYSICS_ENGINE_WINDOW_CLASS"), mWC.hInstance);
}