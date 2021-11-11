#include "../../../include/core/platform/Win32ApplicationWindow.h"
#include "../../../include/core/Input.h"

#include <tchar.h>
#include <algorithm>

using namespace PhysicsEngine;

static bool sIsMinimized = false;
static int sWidth = 1920;
static int sHeight = 1080;

KeyCode getKeyCode(WPARAM wParam, LPARAM lParam)
{
    switch (wParam)
    {
    case VK_BACK:
        return KeyCode::Backspace;
    case VK_TAB:
        return KeyCode::Tab;
    case VK_CLEAR:
        return KeyCode::Clear;
    case VK_RETURN:
        return KeyCode::Enter;
    case VK_SHIFT:
        return ((lParam & 0x01000000) != 0) ? KeyCode::RShift : KeyCode::LShift;
    case VK_CONTROL:
        return ((lParam & 0x01000000) != 0) ? KeyCode::RCtrl : KeyCode::LCtrl;
    case VK_MENU:
        return KeyCode::Menu;
    case VK_PAUSE:
        return KeyCode::Pause;
    case VK_CAPITAL:
        return KeyCode::CapsLock;
    case VK_ESCAPE:
        return KeyCode::Escape;
    case VK_SPACE:
        return KeyCode::Space;
    case VK_LEFT:
        return KeyCode::Left;
    case VK_UP:
        return KeyCode::Up;
    case VK_RIGHT:
        return KeyCode::Right;
    case VK_DOWN:
        return KeyCode::Down;   
    case VK_SNAPSHOT:
        return KeyCode::PrintScreen;
    case VK_INSERT:
        return KeyCode::Insert;
    case VK_DELETE:
        return KeyCode::Delete;
    case VK_HELP:
        return KeyCode::Help;
    case 0x30:
        return KeyCode::Key0;
    case 0x31:
        return KeyCode::Key1;
    case 0x32:
        return KeyCode::Key2;
    case 0x33:
        return KeyCode::Key3;
    case 0x34:
        return KeyCode::Key4;
    case 0x35:
        return KeyCode::Key5;
    case 0x36:
        return KeyCode::Key6;
    case 0x37:
        return KeyCode::Key7;
    case 0x38:
        return KeyCode::Key8;
    case 0x39:
        return KeyCode::Key9;
    case 0x41:
        return KeyCode::A;
    case 0x42:
        return KeyCode::B;
    case 0x43:
        return KeyCode::C;
    case 0x44:
        return KeyCode::D;
    case 0x45:
        return KeyCode::E;
    case 0x46:
        return KeyCode::F;
    case 0x47:
        return KeyCode::G;
    case 0x48:
        return KeyCode::H;
    case 0x49:
        return KeyCode::I;
    case 0x4A:
        return KeyCode::J;
    case 0x4B:
        return KeyCode::K;
    case 0x4C:
        return KeyCode::L;
    case 0x4D:
        return KeyCode::M;
    case 0x4E:
        return KeyCode::N;
    case 0x4F:
        return KeyCode::O;
    case 0x50:
        return KeyCode::P;
    case 0x51:
        return KeyCode::Q;
    case 0x52:
        return KeyCode::R;
    case 0x53:
        return KeyCode::S;
    case 0x54:
        return KeyCode::T;
    case 0x55:
        return KeyCode::U;
    case 0x56:
        return KeyCode::V;
    case 0x57:
        return KeyCode::W;
    case 0x58:
        return KeyCode::X;
    case 0x59:
        return KeyCode::Y;
    case 0x5A:
        return KeyCode::Z;
    case VK_NUMPAD0:
        return KeyCode::NumPad0;
    case VK_NUMPAD1:
        return KeyCode::NumPad1;
    case VK_NUMPAD2:
        return KeyCode::NumPad2;
    case VK_NUMPAD3:
        return KeyCode::NumPad3;
    case VK_NUMPAD4:
        return KeyCode::NumPad4;
    case VK_NUMPAD5:
        return KeyCode::NumPad5;
    case VK_NUMPAD6:
        return KeyCode::NumPad6;
    case VK_NUMPAD7:
        return KeyCode::NumPad7;
    case VK_NUMPAD8:
        return KeyCode::NumPad8;
    case VK_NUMPAD9:
        return KeyCode::NumPad9;
    case VK_MULTIPLY:
        return KeyCode::NumPadMultiply;
    case VK_ADD:
        return KeyCode::NumPadAdd;
    case VK_SUBTRACT:
        return KeyCode::NumPadSubtract;
    case VK_DIVIDE:
        return KeyCode::NumPadDivide;
    case VK_F1:
        return KeyCode::F1;
    case VK_F2:
        return KeyCode::F2;
    case VK_F3:
        return KeyCode::F3;
    case VK_F4:
        return KeyCode::F4;
    case VK_F5:
        return KeyCode::F5;
    case VK_F6:
        return KeyCode::F6;
    case VK_F7:
        return KeyCode::F7;
    case VK_F8:
        return KeyCode::F8;
    case VK_F9:
        return KeyCode::F9;
    case VK_F10:
        return KeyCode::F10;
    case VK_F11:
        return KeyCode::F11;
    case VK_F12:
        return KeyCode::F12;
    default:
        return KeyCode::Invalid;
    }
}

// define in application
extern LRESULT Application_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    Input &input = getInput();

    // Update input (but do not consume messages to allow application to also see and use them)
    switch (msg)
    {
    case WM_LBUTTONDOWN: case WM_LBUTTONDBLCLK:
    case WM_RBUTTONDOWN: case WM_RBUTTONDBLCLK:
    case WM_MBUTTONDOWN: case WM_MBUTTONDBLCLK:
    case WM_XBUTTONDOWN: case WM_XBUTTONDBLCLK:
    {
        MouseButton button = MouseButton::LButton;
        if (msg == WM_LBUTTONDOWN || msg == WM_LBUTTONDBLCLK) { button = MouseButton::LButton; }
        if (msg == WM_RBUTTONDOWN || msg == WM_RBUTTONDBLCLK) { button = MouseButton::RButton; }
        if (msg == WM_MBUTTONDOWN || msg == WM_MBUTTONDBLCLK) { button = MouseButton::MButton; }
        if (msg == WM_XBUTTONDOWN || msg == WM_XBUTTONDBLCLK) { button = (GET_XBUTTON_WPARAM(wParam) == XBUTTON1) ? MouseButton::Alt0Button : MouseButton::Alt1Button; }
        input.mMouseButtonIsDown[static_cast<int>(button)] = 1;

        break;
    }
    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_MBUTTONUP:
    case WM_XBUTTONUP:
    {
        MouseButton button = MouseButton::LButton;
        if (msg == WM_LBUTTONUP) { button = MouseButton::LButton; }
        if (msg == WM_RBUTTONUP) { button = MouseButton::RButton; }
        if (msg == WM_MBUTTONUP) { button = MouseButton::MButton; }
        if (msg == WM_XBUTTONUP) { button = (GET_XBUTTON_WPARAM(wParam) == XBUTTON1) ? MouseButton::Alt0Button : MouseButton::Alt1Button; }
        input.mMouseButtonIsDown[static_cast<int>(button)] = 0;
        break;
    }
    case WM_MOUSEWHEEL:
        input.mMouseDelta += (float)GET_WHEEL_DELTA_WPARAM(wParam) / (float)WHEEL_DELTA;
        break;
    case WM_MOUSEHWHEEL:
        input.mMouseDeltaH += (float)GET_WHEEL_DELTA_WPARAM(wParam) / (float)WHEEL_DELTA;
        break;
    case WM_MOUSEMOVE:
        POINT point;
        if (GetCursorPos(&point))
        {
            // Top left hand corner is (0, 0) on windows
            POINT tl;
            tl.x = 0;
            tl.y = 0;
            ClientToScreen(hWnd, &tl);
            POINT br;
            br.x = sWidth;
            br.y = sHeight;
            ClientToScreen(hWnd, &br);

            // Mouse position stored with (0, 0) at bottom left and (width, height) at top right
            input.mMousePosX = std::min(std::max(point.x - tl.x, (LONG)0), (LONG)sWidth);
            input.mMousePosY = sHeight - std::min(std::max(point.y - tl.y, (LONG)0), (LONG)sHeight);
        }
        break;
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
        if (wParam < 256) 
        {
            input.mKeyIsDown[static_cast<int>(getKeyCode(wParam, lParam))] = 1;
        }
        break;
    case WM_KEYUP:
    case WM_SYSKEYUP:
        if (wParam < 256) 
        {
            input.mKeyIsDown[static_cast<int>(getKeyCode(wParam, lParam))] = 0;
        }
        break;
    //case WM_CHAR:
    //    if (wParam > 0 && wParam < 0x10000) 
    //    {
    //        std::cout << "BBB" << std::endl;
    //    }
    //    //io.AddInputCharacterUTF16((unsigned short)wParam);
    //    break;
    }

    // Call application win proc handler
    if (Application_WndProcHandler(hWnd, msg, wParam, lParam))
        return 0;

    // Window resize and quit messages
    switch (msg)
    {
    case WM_SIZING: 
        {
        RECT rect;
        if (GetClientRect(hWnd, &rect))
        {
            sWidth = rect.right - rect.left;
            sHeight = rect.bottom - rect.top;
        }
        return true;
        }
    case WM_SIZE:
        RECT rect;
        if (GetClientRect(hWnd, &rect))
        {
            sWidth = rect.right - rect.left;
            sHeight = rect.bottom - rect.top;
        }
        if (wParam == SIZE_MINIMIZED)
        {
            sIsMinimized = true;
            return 0;
        }
        else if (wParam == SIZE_MAXIMIZED)
        {
            sIsMinimized = false;
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

    // Use default win proc handler to consume any remaining messages
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
    Input &input = getInput();
    for (int i = 0; i < static_cast<int>(KeyCode::Count); i++)
    {
        input.mKeyWasDown[i] = input.mKeyIsDown[i];
    }

    for (int i = 0; i < static_cast<int>(MouseButton::Count); i++)
    {
        input.mMouseButtonWasDown[i] = input.mMouseButtonIsDown[i];
    }

    input.mMouseDelta = 0.0f;
    input.mMouseDeltaH = 0.0f;

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
    return sWidth;
}

int Win32ApplicationWindow::getHeight() const
{
    return sHeight;
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
    return sIsMinimized;
}

void Win32ApplicationWindow::init(const std::string& title, int width, int height)
{
    mTitle = title;
    sWidth = width;
    sHeight = height;

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
        CW_USEDEFAULT, sWidth, sHeight, 0, 0, mWC.hInstance, 0);

    // Show the window
    ShowWindow(mWindow, SW_SHOWDEFAULT);
    UpdateWindow(mWindow);
}

void Win32ApplicationWindow::cleanup()
{
    DestroyWindow(mWindow);
    UnregisterClass(_T("PHYSICS_ENGINE_WINDOW_CLASS"), mWC.hInstance);
}