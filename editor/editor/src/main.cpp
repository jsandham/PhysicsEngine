#pragma warning(disable:4996)
#define _CRTDBG_MAP_ALLOC  

#include <stdlib.h>  
#include <crtdbg.h> 

#include <windows.h>
#include <windowsx.h>
#include <tchar.h>
#include <xinput.h>
#include <glew-2.1.0/GL/glew.h>
#include <gl/gl.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>

#ifndef WGL_EXT_extensions_string
#define WGL_EXT_extensions_string 1
#ifdef WGL_WGLEXT_PROTOTYPES
extern const char * WINAPI wglGetExtensionsStringEXT(void);
#endif /* WGL_WGLEXT_PROTOTYPES */
typedef const char * (WINAPI * PFNWGLGETEXTENSIONSSTRINGEXTPROC) (void);
#endif

#ifndef WGL_EXT_swap_control
#define WGL_EXT_swap_control 1
#ifdef WGL_WGLEXT_PROTOTYPES
extern BOOL WINAPI wglSwapIntervalEXT(int);
extern int WINAPI wglGetSwapIntervalEXT(void);
#endif /* WGL_WGLEXT_PROTOTYPES */
typedef BOOL(WINAPI * PFNWGLSWAPINTERVALEXTPROC) (int interval);
typedef int (WINAPI * PFNWGLGETSWAPINTERVALEXTPROC) (void);
#endif

#include "../include/Editor.h"

using namespace PhysicsEditor;


// =============================================================================
//                                  DEFINES/MACROS
// =============================================================================

// =============================================================================
//                               GLOBAL VARIABLES
// =============================================================================
HGLRC   g_GLRenderContext;
HDC     g_HDCDeviceContext;
HWND    g_hwnd;
PFNWGLSWAPINTERVALEXTPROC       wglSwapIntervalEXT;
PFNWGLGETSWAPINTERVALEXTPROC    wglGetSwapIntervalEXT;
int     g_display_w = 1024;
int     g_display_h = 1024;
Input input;

// =============================================================================
//                             FOWARD DECLARATIONS
// =============================================================================
void CreateGlContext();
void SetCurrentContext();
bool SetSwapInterval(int interval); //0 - No Interval, 1 - Sync whit VSYNC, n - n times Sync with VSYNC
bool WGLExtensionSupported(const char *extension_name);
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// =============================================================================
//                            CORE MAIN FUNCTIONS
// =============================================================================
//
// Application Entry
//------------------------------------------------------------------------------
int WINAPI wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow){
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
	g_hwnd = CreateWindowEx(0, wc.lpszClassName, _T("PhysicsEngine"), WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, 1024, 1024, 0, 0, hInstance, 0);

	// Show the window
	ShowWindow(g_hwnd, SW_SHOWDEFAULT);
	UpdateWindow(g_hwnd);

	//Prepare OpenGlContext
	CreateGlContext();
	SetSwapInterval(1);
	glewInit();

	Editor editor;

	// initialize editor
	editor.init(g_hwnd, g_display_w, g_display_h);

	// Main loop
	MSG msg;
	ZeroMemory(&msg, sizeof(msg));
	while (msg.message != WM_QUIT)
	{
		// Poll and handle messages (inputs, window resize, etc.)
		// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
		// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
		// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
		// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
		if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
			continue;
		}

		wglMakeCurrent(g_HDCDeviceContext, g_GLRenderContext);

		editor.render(g_hwnd == GetActiveWindow());

		if (editor.getCurrentProjectPath() != ""){
			SetWindowTextA(g_hwnd, ("Physics Engine - " + editor.getCurrentProjectPath()).c_str());
		}
		else {
			SetWindowTextA(g_hwnd, "Physics Engine");
		}

		SwapBuffers(g_HDCDeviceContext);

		if (editor.isQuitCalled()){
			break;
		}
	}

	// Cleanup
	editor.cleanUp();

	wglDeleteContext(g_GLRenderContext);

	DestroyWindow(g_hwnd);
	UnregisterClass(_T("NCUI"), wc.hInstance);

	return 0;
}

extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){
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
		PostQuitMessage(0);
		return 0;
	}

	return DefWindowProc(hWnd, msg, wParam, lParam);
}

void CreateGlContext(){
	PIXELFORMATDESCRIPTOR desiredPixelFormat = {};
	desiredPixelFormat.nSize = sizeof(desiredPixelFormat);
	desiredPixelFormat.nVersion = 1;
	desiredPixelFormat.dwFlags = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER;
	desiredPixelFormat.cColorBits = 32;
	desiredPixelFormat.cAlphaBits = 8;
	desiredPixelFormat.iLayerType = PFD_MAIN_PLANE;

	g_HDCDeviceContext = GetDC(g_hwnd);

	int suggestedPixelFormatIndex = ChoosePixelFormat(g_HDCDeviceContext, &desiredPixelFormat);

	PIXELFORMATDESCRIPTOR suggestedPixelFormat;
	DescribePixelFormat(g_HDCDeviceContext, suggestedPixelFormatIndex, sizeof(suggestedPixelFormat), &suggestedPixelFormat);
	SetPixelFormat(g_HDCDeviceContext, suggestedPixelFormatIndex, &suggestedPixelFormat);

	g_GLRenderContext = wglCreateContext(g_HDCDeviceContext);
	if (!wglMakeCurrent(g_HDCDeviceContext, g_GLRenderContext))
	{
		return;
	}

	//PIXELFORMATDESCRIPTOR pfd =
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

	//g_HDCDeviceContext = GetDC(g_hwnd);

	//int pixelFormal = ChoosePixelFormat(g_HDCDeviceContext, &pfd);
	//SetPixelFormat(g_HDCDeviceContext, pixelFormal, &pfd);
	//g_GLRenderContext = wglCreateContext(g_HDCDeviceContext);
	//wglMakeCurrent(g_HDCDeviceContext, g_GLRenderContext);
}

bool SetSwapInterval(int interval){
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

//Got from https://stackoverflow.com/questions/589064/how-to-enable-vertical-sync-in-opengl/589232
bool WGLExtensionSupported(const char *extension_name){
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
//KeyCode GetKeyCode(unsigned int vKCode)
//{
//	KeyCode keyCode;
//	switch (vKCode)
//	{
//	case 'A':{ keyCode = KeyCode::A; break; }
//	case 'B':{ keyCode = KeyCode::B; break; }
//	case 'C':{ keyCode = KeyCode::C; break; }
//	case 'D':{ keyCode = KeyCode::D; break; }
//	case 'E':{ keyCode = KeyCode::E; break; }
//	case 'F':{ keyCode = KeyCode::F; break; }
//	case 'G':{ keyCode = KeyCode::G; break; }
//	case 'H':{ keyCode = KeyCode::H; break; }
//	case 'I':{ keyCode = KeyCode::I; break; }
//	case 'J':{ keyCode = KeyCode::J; break; }
//	case 'K':{ keyCode = KeyCode::K; break; }
//	case 'L':{ keyCode = KeyCode::L; break; }
//	case 'M':{ keyCode = KeyCode::M; break; }
//	case 'N':{ keyCode = KeyCode::N; break; }
//	case 'O':{ keyCode = KeyCode::O; break; }
//	case 'P':{ keyCode = KeyCode::P; break; }
//	case 'Q':{ keyCode = KeyCode::Q; break; }
//	case 'R':{ keyCode = KeyCode::R; break; }
//	case 'S':{ keyCode = KeyCode::S; break; }
//	case 'T':{ keyCode = KeyCode::T; break; }
//	case 'U':{ keyCode = KeyCode::U; break; }
//	case 'V':{ keyCode = KeyCode::V; break; }
//	case 'W':{ keyCode = KeyCode::W; break; }
//	case 'X':{ keyCode = KeyCode::X; break; }
//	case 'Y':{ keyCode = KeyCode::Y; break; }
//	case 'Z':{ keyCode = KeyCode::Z; break; }
//	case VK_RETURN:{ keyCode = KeyCode::Enter; break; }
//	case VK_UP:{ keyCode = KeyCode::Up; break; }
//	case VK_DOWN:{ keyCode = KeyCode::Down; break; }
//	case VK_LEFT:{ keyCode = KeyCode::Left; break; }
//	case VK_RIGHT:{ keyCode = KeyCode::Right; break; }
//	case VK_SPACE:{ keyCode = KeyCode::Space; break; }
//	case VK_LSHIFT:{ keyCode = KeyCode::LShift; break; }
//	case VK_RSHIFT:{ keyCode = KeyCode::RShift; break; }
//	case VK_TAB:{ keyCode = KeyCode::Tab; break; }
//	case VK_BACK:{ keyCode = KeyCode::Backspace; break; }
//	case VK_CAPITAL:{ keyCode = KeyCode::CapsLock; break; }
//	case VK_LCONTROL:{ keyCode = KeyCode::LCtrl; break; }
//	case VK_RCONTROL:{ keyCode = KeyCode::RCtrl; break; }
//	case VK_ESCAPE:{ keyCode = KeyCode::Backspace; break; }
//	case VK_NUMPAD0:{ keyCode = KeyCode::NumPad0; break; }
//	case VK_NUMPAD1:{ keyCode = KeyCode::NumPad1; break; }
//	case VK_NUMPAD2:{ keyCode = KeyCode::NumPad2; break; }
//	case VK_NUMPAD3:{ keyCode = KeyCode::NumPad3; break; }
//	case VK_NUMPAD4:{ keyCode = KeyCode::NumPad4; break; }
//	case VK_NUMPAD5:{ keyCode = KeyCode::NumPad5; break; }
//	case VK_NUMPAD6:{ keyCode = KeyCode::NumPad6; break; }
//	case VK_NUMPAD7:{ keyCode = KeyCode::NumPad7; break; }
//	case VK_NUMPAD8:{ keyCode = KeyCode::NumPad8; break; }
//	case VK_NUMPAD9:{ keyCode = KeyCode::NumPad9; break; }
//	default:{ keyCode = KeyCode::Invalid; break; }
//	}
//
//	return keyCode;
//}

