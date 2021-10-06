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

    mContext = ApplicationGraphicsContext::createApplicationGraphicsContext(static_cast<void*>(g_hwnd));
}

Win32ApplicationWindow::~Win32ApplicationWindow()
{
	cleanup();

    delete mContext;
}

void Win32ApplicationWindow::update()
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

    mContext->update();
}

int Win32ApplicationWindow::getWidth() const
{
	return mWidth;
}

int Win32ApplicationWindow::getHeight() const
{
	return mHeight;
}

void Win32ApplicationWindow::init(const std::string& title, int width, int height)
{
    mTitle = title;
    mWidth = width;
    mHeight = height;

    wc = { 0 };
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(0);
    wc.hbrBackground = (HBRUSH)(COLOR_BACKGROUND);
    wc.lpszClassName = _T("NCUI");
    wc.style = CS_OWNDC;
    if (!RegisterClass(&wc))
        return;
    g_hwnd = CreateWindowEx(0, wc.lpszClassName, _T(mTitle.c_str()), WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT,
        CW_USEDEFAULT, mWidth, mHeight, 0, 0, wc.hInstance, 0);

    // Show the window
    ShowWindow(g_hwnd, SW_SHOWDEFAULT);
    UpdateWindow(g_hwnd);

    prevActiveWindow = NULL;
    activeWindow = NULL;

    ZeroMemory(&msg, sizeof(msg));
}

void Win32ApplicationWindow::cleanup()
{
    DestroyWindow(g_hwnd);
    UnregisterClass(_T("NCUI"), wc.hInstance);
}