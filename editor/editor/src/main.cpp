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
int     g_display_w = 800;
int     g_display_h = 600;

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
	g_hwnd = CreateWindowEx(0, wc.lpszClassName, _T("PhysicsEngine"), WS_OVERLAPPEDWINDOW | WS_VISIBLE, 0, 0, 1920, 1080, 0, 0, hInstance, 0);

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

		editor.render();

		wglMakeCurrent(g_HDCDeviceContext, g_GLRenderContext);

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
		return true;

	switch (msg)
	{
	case WM_SIZE:
		if (wParam != SIZE_MINIMIZED)
		{
			g_display_w = (UINT)LOWORD(lParam);
			g_display_h = (UINT)HIWORD(lParam);
		}
		return 0;
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

	PIXELFORMATDESCRIPTOR pfd =
	{
		sizeof(PIXELFORMATDESCRIPTOR),
		1,
		PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,    //Flags
		PFD_TYPE_RGBA,        // The kind of framebuffer. RGBA or palette.
		32,                   // Colordepth of the framebuffer.
		0, 0, 0, 0, 0, 0,
		0,
		0,
		0,
		0, 0, 0, 0,
		24,                   // Number of bits for the depthbuffer
		8,                    // Number of bits for the stencilbuffer
		0,                    // Number of Aux buffers in the framebuffer.
		PFD_MAIN_PLANE,
		0,
		0, 0, 0
	};

	g_HDCDeviceContext = GetDC(g_hwnd);

	int pixelFormal = ChoosePixelFormat(g_HDCDeviceContext, &pfd);
	SetPixelFormat(g_HDCDeviceContext, pixelFormal, &pfd);
	g_GLRenderContext = wglCreateContext(g_HDCDeviceContext);
	wglMakeCurrent(g_HDCDeviceContext, g_GLRenderContext);
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
//
//static std::string current_working_directory()
//{
//	char* cwd = _getcwd(0, 0); // **** microsoft specific ****
//	std::string working_directory(cwd);
//	std::free(cwd);
//	return working_directory;
//}

//static std::vector<std::string> get_all_files_names_within_folder(std::string folder, std::string extension)
//{
//	std::vector<std::string> names;
//	std::string search_path = folder + "/*.*";
//	WIN32_FIND_DATA fd;
//	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
//	if (hFind != INVALID_HANDLE_VALUE) {
//		do {
//			// read all (real) files in current folder
//			// , delete '!' read other 2 default folder . and ..
//			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
//
//				std::string file = fd.cFileName;
//				if (file.substr(file.find_last_of(".") + 1) == extension) {
//					names.push_back(folder + file);
//				}
//				//names.push_back(fd.cFileName);
//			}
//		} while (::FindNextFile(hFind, &fd));
//		::FindClose(hFind);
//	}
//	return names;
//}


























//
//
//#pragma warning(disable:4996)
//#define _CRTDBG_MAP_ALLOC  
//
//#include <stdlib.h>  
//#include <crtdbg.h> 
//
//#include <windows.h>
//#include <windowsx.h>
//#include <tchar.h>
//#include <xinput.h>
//#include <glew-2.1.0/GL/glew.h>
//#include <gl/gl.h>
//#include <stdio.h>
//#include <iostream>
//#include <fstream>
//#include <string>
//#include <stdio.h>
//
//#ifndef WGL_EXT_extensions_string
//#define WGL_EXT_extensions_string 1
//#ifdef WGL_WGLEXT_PROTOTYPES
//extern const char * WINAPI wglGetExtensionsStringEXT(void);
//#endif /* WGL_WGLEXT_PROTOTYPES */
//typedef const char * (WINAPI * PFNWGLGETEXTENSIONSSTRINGEXTPROC) (void);
//#endif
//
//#ifndef WGL_EXT_swap_control
//#define WGL_EXT_swap_control 1
//#ifdef WGL_WGLEXT_PROTOTYPES
//extern BOOL WINAPI wglSwapIntervalEXT(int);
//extern int WINAPI wglGetSwapIntervalEXT(void);
//#endif /* WGL_WGLEXT_PROTOTYPES */
//typedef BOOL(WINAPI * PFNWGLSWAPINTERVALEXTPROC) (int interval);
//typedef int (WINAPI * PFNWGLGETSWAPINTERVALEXTPROC) (void);
//#endif
//
//#include "../include/imgui/imgui.h"
//#include "../include/imgui/imgui_impl_win32.h"
//
//#include "core/Scene.h"
//#include "core/Asset.h"
//#include "core/Input.h"
//#include "core/Time.h"
//#include "core/Log.h"
//#include "core/WorldManager.h"
//
//using namespace PhysicsEngine;
//
//static bool running;
//
//Time time;
//Input input;
//
//size_t frameCount = 0;
//LARGE_INTEGER lastCounter;
//LARGE_INTEGER perfCounterFrequencyResult;
//unsigned long long lastCycleCount;
//long long perfCounterFrequency;
//
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
//
//bool WGLExtensionSupported(const char *extension_name)
//{
//	// this is pointer to function which returns pointer to string with list of all wgl extensions
//	PFNWGLGETEXTENSIONSSTRINGEXTPROC _wglGetExtensionsStringEXT = NULL;
//
//	// determine pointer to wglGetExtensionsStringEXT function
//	_wglGetExtensionsStringEXT = (PFNWGLGETEXTENSIONSSTRINGEXTPROC)wglGetProcAddress("wglGetExtensionsStringEXT");
//
//	if (strstr(_wglGetExtensionsStringEXT(), extension_name) == NULL)
//	{
//		// string was not found
//		return false;
//	}
//
//	// extension is supported
//	return true;
//}
//
//bool Win32InitOpenGL(HWND window)
//{
//	HDC windowDC = GetDC(window);
//
//	PIXELFORMATDESCRIPTOR desiredPixelFormat = {};
//	desiredPixelFormat.nSize = sizeof(desiredPixelFormat);
//	desiredPixelFormat.nVersion = 1;
//	desiredPixelFormat.dwFlags = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER;
//	desiredPixelFormat.cColorBits = 32;
//	desiredPixelFormat.cAlphaBits = 8;
//	desiredPixelFormat.iLayerType = PFD_MAIN_PLANE;
//
//	int suggestedPixelFormatIndex = ChoosePixelFormat(windowDC, &desiredPixelFormat);
//
//	PIXELFORMATDESCRIPTOR suggestedPixelFormat;
//	DescribePixelFormat(windowDC, suggestedPixelFormatIndex, sizeof(suggestedPixelFormat), &suggestedPixelFormat);
//	SetPixelFormat(windowDC, suggestedPixelFormatIndex, &suggestedPixelFormat);
//
//	HGLRC openGLRC = wglCreateContext(windowDC);
//	//wglCreateContextAttrib();
//	if (!wglMakeCurrent(windowDC, openGLRC))
//	{
//		OutputDebugStringA("ERROR: OPENGL INIT FALIED\n");
//		return false;
//
//	}
//
//	ReleaseDC(window, windowDC);
//
//	glewExperimental = GL_TRUE;
//	GLenum err = glewInit();
//	if (err != GLEW_OK){
//		OutputDebugStringA("ERROR: COULD NOT INITIALIZE GLEW\n");
//
//		char buffer[256];
//		sprintf_s(buffer, "glew error code %d\n", err);
//		OutputDebugStringA(buffer);
//		return false;
//	}
//	else
//	{
//		OutputDebugStringA("INITIALIZED GLEW SUCCESSFULLY\n");
//	}
//
//	PFNWGLSWAPINTERVALEXTPROC       wglSwapIntervalEXT = NULL;
//	PFNWGLGETSWAPINTERVALEXTPROC    wglGetSwapIntervalEXT = NULL;
//
//	if (WGLExtensionSupported("WGL_EXT_swap_control"))
//	{
//		std::cout << "vsync extension supported" << std::endl;
//
//		// Extension is supported, init pointers.
//		wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
//
//		// this is another function from WGL_EXT_swap_control extension
//		wglGetSwapIntervalEXT = (PFNWGLGETSWAPINTERVALEXTPROC)wglGetProcAddress("wglGetSwapIntervalEXT");
//
//		wglSwapIntervalEXT(0);//vsync On: 1 Off: 0
//	}
//
//	OutputDebugStringA("CUDA DEVICE SELECTED\n");
//
//	return true;
//}
//
//void Win32UpdateWindow(HDC windowDC, int x, int y, int width, int height)
//{
//	SwapBuffers(windowDC);
//}
//
//LRESULT CALLBACK MainWindowCallback(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
//{
//	LRESULT result = 0;
//
//	switch (uMsg)
//	{
//	case WM_SIZE:
//	{
//		OutputDebugStringA("WM_SIZE CALLED\n");
//		break;
//	}
//	case WM_CLOSE:
//		running = false;
//		std::cout << "EXIT CALLED" << std::endl;
//		PostQuitMessage(0);
//		break;
//	case WM_DESTROY:
//	{
//		running = false;
//		std::cout << "EXIT CALLED" << std::endl;
//		break;
//	}
//	case WM_ACTIVATEAPP:
//	{
//		break;
//	}
//	case WM_SYSKEYDOWN:
//	case WM_KEYDOWN:
//	case WM_SYSKEYUP:
//	case WM_KEYUP:
//	{
//		unsigned int vKCode = (unsigned int)wParam;
//		bool wasDown = ((lParam & (1 << 30)) != 0);
//		bool isDown = ((lParam & (1 << 31)) == 0);
//
//		KeyCode keyCode = GetKeyCode(vKCode);
//		input.keyIsDown[(int)keyCode] = isDown;
//		input.keyWasDown[(int)keyCode] = wasDown;
//		break;
//	}
//	case WM_LBUTTONDOWN:
//	{
//		input.mouseButtonIsDown[(int)LButton] = true;
//		input.mouseButtonWasDown[(int)LButton] = false;
//		break;
//	}
//	case WM_MBUTTONDOWN:
//	{
//		input.mouseButtonIsDown[(int)MButton] = true;
//		input.mouseButtonWasDown[(int)MButton] = false;
//		break;
//	}
//	case WM_RBUTTONDOWN:
//	{
//		input.mouseButtonIsDown[(int)RButton] = true;
//		input.mouseButtonWasDown[(int)RButton] = false;
//		break;
//	}
//	case WM_LBUTTONUP:
//	{
//		input.mouseButtonIsDown[(int)LButton] = false;
//		input.mouseButtonWasDown[(int)LButton] = true;
//		break;
//	}
//	case WM_MBUTTONUP:
//	{
//		input.mouseButtonIsDown[(int)MButton] = false;
//		input.mouseButtonWasDown[(int)MButton] = true;
//		break;
//	}
//	case WM_RBUTTONUP:
//	{
//		input.mouseButtonIsDown[(int)RButton] = false;
//		input.mouseButtonWasDown[(int)RButton] = true;
//		break;
//	}
//	case WM_MOUSEMOVE:
//	{
//		int x = GET_X_LPARAM(lParam);
//		int y = GET_Y_LPARAM(lParam);
//		input.mousePosX = x;
//		input.mousePosY = y;
//		break;
//	}
//	case WM_MOUSEWHEEL:
//	{
//		int delta = GET_WHEEL_DELTA_WPARAM(wParam);
//		input.mouseDelta = delta;
//		break;
//	}
//	case WM_PAINT:
//	{
//		PAINTSTRUCT paint;
//		HDC deviceContext = BeginPaint(hwnd, &paint);
//		int x = paint.rcPaint.left;
//		int y = paint.rcPaint.top;
//		int width = paint.rcPaint.right - paint.rcPaint.left;
//		int height = paint.rcPaint.bottom - paint.rcPaint.top;
//		Win32UpdateWindow(deviceContext, x, y, width, height);
//		EndPaint(hwnd, &paint);
//		break;
//	}
//	default:
//	{
//		result = DefWindowProc(hwnd, uMsg, wParam, lParam);
//		break;
//	}
//	}
//
//	return result;
//}
//
//void CreateConsole()
//{
//	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
//	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
//
//	if (AllocConsole()){
//		freopen("CONIN$", "r", stdin);
//		freopen("CONOUT$", "w", stdout);
//		freopen("CONOUT$", "w", stderr);
//	}
//}
//
//void StartTime()
//{
//	frameCount = 0;
//	QueryPerformanceCounter(&lastCounter);
//	QueryPerformanceFrequency(&perfCounterFrequencyResult);
//	lastCycleCount = __rdtsc();
//	perfCounterFrequency = perfCounterFrequencyResult.QuadPart;
//
//	time.frameCount = 0;
//	time.time = 0.0f;
//	time.deltaTime = 0.0f;
//	time.deltaCycles = 0;
//}
//
//void RecordTime()
//{
//	unsigned long long endCycleCount = __rdtsc();
//	LARGE_INTEGER endCounter;
//	QueryPerformanceCounter(&endCounter);
//
//	unsigned long long cyclesElapsed = endCycleCount - lastCycleCount;
//	long long counterElapsed = endCounter.QuadPart - lastCounter.QuadPart;
//	float megaCyclesPerFrame = ((float)cyclesElapsed / (1000.0f * 1000.0f));
//	float milliSecPerFrame = ((1000.0f*(float)counterElapsed) / (float)perfCounterFrequency);
//
//	lastCycleCount = endCycleCount;
//	lastCounter = endCounter;
//	frameCount++;
//
//	time.frameCount = frameCount;
//	time.time = (1000.0f * (float)lastCounter.QuadPart) / ((float)perfCounterFrequency);
//	time.deltaTime = ((1000.0f*(float)counterElapsed) / (float)perfCounterFrequency);
//	time.deltaCycles = (size_t)cyclesElapsed;
//}
//
//
//int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
//{
//	CreateConsole();
//
//	WNDCLASS windowClass = {};
//	windowClass.style = CS_HREDRAW | CS_VREDRAW;
//	windowClass.lpfnWndProc = MainWindowCallback;
//	windowClass.hInstance = hInstance;
//	windowClass.lpszClassName = _T("PhysicsEngineWindowClass");
//
//	if (!RegisterClass(&windowClass)){
//		return 0;
//	}
//
//	HWND windowHandle = CreateWindowEx(0,
//		windowClass.lpszClassName,
//		_T("PhysicsEngine"),
//		WS_OVERLAPPEDWINDOW | WS_VISIBLE,
//		CW_USEDEFAULT,
//		CW_USEDEFAULT,
//		1024,
//		1024,
//		0,
//		0,
//		hInstance,
//		0);
//
//	if (!windowHandle){
//		return 0;
//	}
//
//	IMGUI_CHECKVERSION();
//	ImGui::CreateContext();
//	//ImGui::SetCurrentContext();
//
//	if (ImGui_ImplWin32_Init(windowHandle))
//	{
//		std::cout << "AAAAAAAA" << std::endl;
//		//return 0;
//	}
//
//	if (!Win32InitOpenGL(windowHandle)){
//		return 0;
//	}
//
//	StartTime();
//
//	running = true;
//
//	while (running)
//	{
//		MSG message;
//		while (PeekMessage(&message, 0, 0, 0, PM_REMOVE))
//		{
//			if (message.message == WM_QUIT){ running = false; }
//
//			TranslateMessage(&message);
//			DispatchMessage(&message);
//		}
//
//		for (DWORD controllerIndex = 0; controllerIndex < XUSER_MAX_COUNT; controllerIndex++)
//		{
//			XINPUT_STATE ControllerState;
//			if (XInputGetState(controllerIndex, &ControllerState) == ERROR_SUCCESS){
//				XINPUT_GAMEPAD *Pad = &ControllerState.Gamepad;
//
//				input.xboxButtonIsDown[XboxButton::LeftDPad] = (Pad->wButtons & XINPUT_GAMEPAD_DPAD_LEFT);
//				input.xboxButtonIsDown[XboxButton::RightDPad] = (Pad->wButtons & XINPUT_GAMEPAD_DPAD_RIGHT);
//				input.xboxButtonIsDown[XboxButton::UpDPad] = (Pad->wButtons & XINPUT_GAMEPAD_DPAD_UP);
//				input.xboxButtonIsDown[XboxButton::DownDPad] = (Pad->wButtons & XINPUT_GAMEPAD_DPAD_DOWN);
//				input.xboxButtonIsDown[XboxButton::Start] = (Pad->wButtons & XINPUT_GAMEPAD_START);
//				input.xboxButtonIsDown[XboxButton::Back] = (Pad->wButtons & XINPUT_GAMEPAD_BACK);
//				input.xboxButtonIsDown[XboxButton::LeftThumb] = (Pad->wButtons & XINPUT_GAMEPAD_LEFT_THUMB);
//				input.xboxButtonIsDown[XboxButton::RightThumb] = (Pad->wButtons & XINPUT_GAMEPAD_RIGHT_THUMB);
//				input.xboxButtonIsDown[XboxButton::LeftShoulder] = (Pad->wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER);
//				input.xboxButtonIsDown[XboxButton::RightShoulder] = (Pad->wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER);
//				input.xboxButtonIsDown[XboxButton::AButton] = (Pad->wButtons & XINPUT_GAMEPAD_A);
//				input.xboxButtonIsDown[XboxButton::BButton] = (Pad->wButtons & XINPUT_GAMEPAD_B);
//				input.xboxButtonIsDown[XboxButton::XButton] = (Pad->wButtons & XINPUT_GAMEPAD_X);
//				input.xboxButtonIsDown[XboxButton::YButton] = (Pad->wButtons & XINPUT_GAMEPAD_Y);
//
//				input.leftStickX = Pad->sThumbLX;
//				input.leftStickY = Pad->sThumbLY;
//				input.rightStickX = Pad->sThumbRX;
//				input.rightStickY = Pad->sThumbRY;
//			}
//			else{
//
//			}
//		}
//
//
//		RedrawWindow(windowHandle, 0, 0, RDW_INVALIDATE);
//
//		RecordTime();
//
//		// input
//		for (int i = 0; i < 3; i++){
//			input.mouseButtonWasDown[i] = input.mouseButtonIsDown[i];
//			//input.buttonIsDown[i] = false;
//		}
//
//		input.mouseDelta = 0;
//
//		for (int i = 0; i < 51; i++){
//			input.keyIsDown[i] = false;
//			input.keyWasDown[i] = false;// input.keyIsDown[i];// false
//		}
//
//		for (int i = 0; i < 14; i++){
//			input.xboxButtonWasDown[i] = input.xboxButtonIsDown[i];
//			input.xboxButtonIsDown[i] = false;
//		}
//	}
//
//	printf("DestroyContext()\n");
//	ImGui::DestroyContext();
//
//	return 0;
//}
//


































//// editor.cpp : Defines the entry point for the application.
////
//
//#include "stdafx.h"
//#include "editor.h"
//
//#define MAX_LOADSTRING 100
//
//// Global Variables:
//HINSTANCE hInst;								// current instance
//TCHAR szTitle[MAX_LOADSTRING];					// The title bar text
//TCHAR szWindowClass[MAX_LOADSTRING];			// the main window class name
//
//// Forward declarations of functions included in this code module:
//ATOM				MyRegisterClass(HINSTANCE hInstance);
//BOOL				InitInstance(HINSTANCE, int);
//LRESULT CALLBACK	WndProc(HWND, UINT, WPARAM, LPARAM);
//INT_PTR CALLBACK	About(HWND, UINT, WPARAM, LPARAM);
//
//int APIENTRY _tWinMain(_In_ HINSTANCE hInstance,
//                     _In_opt_ HINSTANCE hPrevInstance,
//                     _In_ LPTSTR    lpCmdLine,
//                     _In_ int       nCmdShow)
//{
//	UNREFERENCED_PARAMETER(hPrevInstance);
//	UNREFERENCED_PARAMETER(lpCmdLine);
//
// 	// TODO: Place code here.
//	MSG msg;
//	HACCEL hAccelTable;
//
//	// Initialize global strings
//	LoadString(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
//	LoadString(hInstance, IDC_EDITOR, szWindowClass, MAX_LOADSTRING);
//	MyRegisterClass(hInstance);
//
//	// Perform application initialization:
//	if (!InitInstance (hInstance, nCmdShow))
//	{
//		return FALSE;
//	}
//
//	hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_EDITOR));
//
//	// Main message loop:
//	while (GetMessage(&msg, NULL, 0, 0))
//	{
//		if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
//		{
//			TranslateMessage(&msg);
//			DispatchMessage(&msg);
//		}
//	}
//
//	return (int) msg.wParam;
//}
//
//
//
////
////  FUNCTION: MyRegisterClass()
////
////  PURPOSE: Registers the window class.
////
//ATOM MyRegisterClass(HINSTANCE hInstance)
//{
//	WNDCLASSEX wcex;
//
//	wcex.cbSize = sizeof(WNDCLASSEX);
//
//	wcex.style			= CS_HREDRAW | CS_VREDRAW;
//	wcex.lpfnWndProc	= WndProc;
//	wcex.cbClsExtra		= 0;
//	wcex.cbWndExtra		= 0;
//	wcex.hInstance		= hInstance;
//	wcex.hIcon			= LoadIcon(hInstance, MAKEINTRESOURCE(IDI_EDITOR));
//	wcex.hCursor		= LoadCursor(NULL, IDC_ARROW);
//	wcex.hbrBackground	= (HBRUSH)(COLOR_WINDOW+1);
//	wcex.lpszMenuName	= MAKEINTRESOURCE(IDC_EDITOR);
//	wcex.lpszClassName	= szWindowClass;
//	wcex.hIconSm		= LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));
//
//	return RegisterClassEx(&wcex);
//}
//
////
////   FUNCTION: InitInstance(HINSTANCE, int)
////
////   PURPOSE: Saves instance handle and creates main window
////
////   COMMENTS:
////
////        In this function, we save the instance handle in a global variable and
////        create and display the main program window.
////
//BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
//{
//   HWND hWnd;
//
//   hInst = hInstance; // Store instance handle in our global variable
//
//   hWnd = CreateWindow(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
//      CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, NULL, hInstance, NULL);
//
//   if (!hWnd)
//   {
//      return FALSE;
//   }
//
//   ShowWindow(hWnd, nCmdShow);
//   UpdateWindow(hWnd);
//
//   return TRUE;
//}
//
////
////  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
////
////  PURPOSE:  Processes messages for the main window.
////
////  WM_COMMAND	- process the application menu
////  WM_PAINT	- Paint the main window
////  WM_DESTROY	- post a quit message and return
////
////
//LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
//{
//	int wmId, wmEvent;
//	PAINTSTRUCT ps;
//	HDC hdc;
//
//	switch (message)
//	{
//	case WM_COMMAND:
//		wmId    = LOWORD(wParam);
//		wmEvent = HIWORD(wParam);
//		// Parse the menu selections:
//		switch (wmId)
//		{
//		case IDM_ABOUT:
//			DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
//			break;
//		case IDM_EXIT:
//			DestroyWindow(hWnd);
//			break;
//		default:
//			return DefWindowProc(hWnd, message, wParam, lParam);
//		}
//		break;
//	case WM_PAINT:
//		hdc = BeginPaint(hWnd, &ps);
//		// TODO: Add any drawing code here...
//		EndPaint(hWnd, &ps);
//		break;
//	case WM_DESTROY:
//		PostQuitMessage(0);
//		break;
//	default:
//		return DefWindowProc(hWnd, message, wParam, lParam);
//	}
//	return 0;
//}
//
//// Message handler for about box.
//INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
//{
//	UNREFERENCED_PARAMETER(lParam);
//	switch (message)
//	{
//	case WM_INITDIALOG:
//		return (INT_PTR)TRUE;
//
//	case WM_COMMAND:
//		if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
//		{
//			EndDialog(hDlg, LOWORD(wParam));
//			return (INT_PTR)TRUE;
//		}
//		break;
//	}
//	return (INT_PTR)FALSE;
//}
