#pragma warning(disable:4996)
#define _CRTDBG_MAP_ALLOC  

#include <stdlib.h>  
#include <crtdbg.h> 

#include <windows.h>
#include <windowsx.h>
#include <tchar.h>
#include <xinput.h>
#include <GL/glew.h>
#include <gl/gl.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

//#include <cuda.h>
//#include <cudagl.h>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>

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

//#include "../../../include/cuda/cuda_util.h"
//#include "cuda/cuda_util.h"

#include "core/Scene.h"
#include "core/Asset.h"
#include "core/Input.h"
#include "core/Time.h"
#include "core/Log.h"
#include "core/WorldManager.h"

using namespace PhysicsEngine;

static bool running;

Time time;
Input input;

size_t frameCount = 0;
LARGE_INTEGER lastCounter;
LARGE_INTEGER perfCounterFrequencyResult;
unsigned long long lastCycleCount;
long long perfCounterFrequency;

KeyCode GetKeyCode(unsigned int vKCode)
{
	KeyCode keyCode;
	switch (vKCode)
	{
	case 'A':{ keyCode = KeyCode::A; break; }
	case 'B':{ keyCode = KeyCode::B; break; }
	case 'C':{ keyCode = KeyCode::C; break; }
	case 'D':{ keyCode = KeyCode::D; break; }
	case 'E':{ keyCode = KeyCode::E; break; }
	case 'F':{ keyCode = KeyCode::F; break; }
	case 'G':{ keyCode = KeyCode::G; break; }
	case 'H':{ keyCode = KeyCode::H; break; }
	case 'I':{ keyCode = KeyCode::I; break; }
	case 'J':{ keyCode = KeyCode::J; break; }
	case 'K':{ keyCode = KeyCode::K; break; }
	case 'L':{ keyCode = KeyCode::L; break; }
	case 'M':{ keyCode = KeyCode::M; break; }
	case 'N':{ keyCode = KeyCode::N; break; }
	case 'O':{ keyCode = KeyCode::O; break; }
	case 'P':{ keyCode = KeyCode::P; break; }
	case 'Q':{ keyCode = KeyCode::Q; break; }
	case 'R':{ keyCode = KeyCode::R; break; }
	case 'S':{ keyCode = KeyCode::S; break; }
	case 'T':{ keyCode = KeyCode::T; break; }
	case 'U':{ keyCode = KeyCode::U; break; }
	case 'V':{ keyCode = KeyCode::V; break; }
	case 'W':{ keyCode = KeyCode::W; break; }
	case 'X':{ keyCode = KeyCode::X; break; }
	case 'Y':{ keyCode = KeyCode::Y; break; }
	case 'Z':{ keyCode = KeyCode::Z; break; }
	case VK_RETURN:{ keyCode = KeyCode::Enter; break; }
	case VK_UP:{ keyCode = KeyCode::Up; break; }
	case VK_DOWN:{ keyCode = KeyCode::Down; break; }
	case VK_LEFT:{ keyCode = KeyCode::Left; break; }
	case VK_RIGHT:{ keyCode = KeyCode::Right; break; }
	case VK_SPACE:{ keyCode = KeyCode::Space; break; }
	case VK_LSHIFT:{ keyCode = KeyCode::LShift; break; }
	case VK_RSHIFT:{ keyCode = KeyCode::RShift; break; }
	case VK_TAB:{ keyCode = KeyCode::Tab; break; }
	case VK_BACK:{ keyCode = KeyCode::Backspace; break; }
	case VK_CAPITAL:{ keyCode = KeyCode::CapsLock; break; }
	case VK_LCONTROL:{ keyCode = KeyCode::LCtrl; break; }
	case VK_RCONTROL:{ keyCode = KeyCode::RCtrl; break; }
	case VK_ESCAPE:{ keyCode = KeyCode::Backspace; break; }
	case VK_NUMPAD0:{ keyCode = KeyCode::NumPad0; break; }
	case VK_NUMPAD1:{ keyCode = KeyCode::NumPad1; break; }
	case VK_NUMPAD2:{ keyCode = KeyCode::NumPad2; break; }
	case VK_NUMPAD3:{ keyCode = KeyCode::NumPad3; break; }
	case VK_NUMPAD4:{ keyCode = KeyCode::NumPad4; break; }
	case VK_NUMPAD5:{ keyCode = KeyCode::NumPad5; break; }
	case VK_NUMPAD6:{ keyCode = KeyCode::NumPad6; break; }
	case VK_NUMPAD7:{ keyCode = KeyCode::NumPad7; break; }
	case VK_NUMPAD8:{ keyCode = KeyCode::NumPad8; break; }
	case VK_NUMPAD9:{ keyCode = KeyCode::NumPad9; break; }
	default:{ keyCode = KeyCode::Invalid; break; }
	}

	return keyCode;
}

bool WGLExtensionSupported(const char *extension_name)
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

bool Win32InitOpenGL(HWND window)
{
	HDC windowDC = GetDC(window);

	PIXELFORMATDESCRIPTOR desiredPixelFormat = {};
	desiredPixelFormat.nSize = sizeof(desiredPixelFormat);
	desiredPixelFormat.nVersion = 1;
	desiredPixelFormat.dwFlags = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER;
	desiredPixelFormat.cColorBits = 32;
	desiredPixelFormat.cAlphaBits = 8;
	desiredPixelFormat.iLayerType = PFD_MAIN_PLANE;

	int suggestedPixelFormatIndex = ChoosePixelFormat(windowDC, &desiredPixelFormat);

	PIXELFORMATDESCRIPTOR suggestedPixelFormat;
	DescribePixelFormat(windowDC, suggestedPixelFormatIndex, sizeof(suggestedPixelFormat), &suggestedPixelFormat);
	SetPixelFormat(windowDC, suggestedPixelFormatIndex, &suggestedPixelFormat);

	HGLRC openGLRC = wglCreateContext(windowDC);
	//wglCreateContextAttrib();
	if (!wglMakeCurrent(windowDC, openGLRC))
	{
		OutputDebugStringA("ERROR: OPENGL INIT FALIED\n");
		return false;

	}

	ReleaseDC(window, windowDC);

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (err != GLEW_OK){
		OutputDebugStringA("ERROR: COULD NOT INITIALIZE GLEW\n");

		char buffer[256];
		sprintf_s(buffer, "glew error code %d\n", err);
		OutputDebugStringA(buffer);
		return false;
	}
	else
	{
		OutputDebugStringA("INITIALIZED GLEW SUCCESSFULLY\n");
	}

	PFNWGLSWAPINTERVALEXTPROC       wglSwapIntervalEXT = NULL;
	PFNWGLGETSWAPINTERVALEXTPROC    wglGetSwapIntervalEXT = NULL;

	if (WGLExtensionSupported("WGL_EXT_swap_control"))
	{
		std::cout << "vsync extension supported" << std::endl;

		// Extension is supported, init pointers.
		wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");

		// this is another function from WGL_EXT_swap_control extension
		wglGetSwapIntervalEXT = (PFNWGLGETSWAPINTERVALEXTPROC)wglGetProcAddress("wglGetSwapIntervalEXT");

		wglSwapIntervalEXT(0);//vsync On: 1 Off: 0
	}

	/*cudaDeviceProp prop;
	int device;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	gpuErrchk(cudaChooseDevice(&device, &prop));*/
	//gpuErrchk(cudaGLSetGLDevice(device));

	OutputDebugStringA("CUDA DEVICE SELECTED\n");

	return true;
}

void Win32UpdateWindow(HDC windowDC, int x, int y, int width, int height)
{
	SwapBuffers(windowDC);
}

LRESULT CALLBACK MainWindowCallback(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	LRESULT result = 0;

	switch (uMsg)
	{
	case WM_SIZE:
	{
		OutputDebugStringA("WM_SIZE CALLED\n");
		break;
	}
	case WM_CLOSE:
		running = false;
		std::cout << "EXIT CALLED" << std::endl;
		PostQuitMessage(0);
		break;
	case WM_DESTROY:
	{
		running = false;
		std::cout << "EXIT CALLED" << std::endl;
		break;
	}
	case WM_ACTIVATEAPP:
	{
		break;
	}
	case WM_SYSKEYDOWN:
	case WM_KEYDOWN:
	case WM_SYSKEYUP:
	case WM_KEYUP:
	{
		unsigned int vKCode = (unsigned int)wParam;
		bool wasDown = ((lParam & (1 << 30)) != 0);
		bool isDown = ((lParam & (1 << 31)) == 0);

		KeyCode keyCode = GetKeyCode(vKCode);
		input.keyIsDown[(int)keyCode] = isDown;
		input.keyWasDown[(int)keyCode] = wasDown;
		//std::cout << "keyCode: " << keyCode << " is down: " << isDown << " was down: " << wasDown << std::endl;
		break;
	}
	case WM_LBUTTONDOWN:
	{
		input.mouseButtonIsDown[(int)LButton] = true;
		input.mouseButtonWasDown[(int)LButton] = false;
		break;
	}
	case WM_MBUTTONDOWN:
	{
		input.mouseButtonIsDown[(int)MButton] = true;
		input.mouseButtonWasDown[(int)MButton] = false;
		break;
	}
	case WM_RBUTTONDOWN:
	{
		input.mouseButtonIsDown[(int)RButton] = true;
		input.mouseButtonWasDown[(int)RButton] = false;
		break;
	}
	case WM_LBUTTONUP:
	{
		input.mouseButtonIsDown[(int)LButton] = false;
		input.mouseButtonWasDown[(int)LButton] = true;
		break;
	}
	case WM_MBUTTONUP:
	{
		input.mouseButtonIsDown[(int)MButton] = false;
		input.mouseButtonWasDown[(int)MButton] = true;
		break;
	}
	case WM_RBUTTONUP:
	{
		input.mouseButtonIsDown[(int)RButton] = false;
		input.mouseButtonWasDown[(int)RButton] = true;
		break;
	}
	case WM_MOUSEMOVE:
	{
		int x = GET_X_LPARAM(lParam);
		int y = GET_Y_LPARAM(lParam);
		input.mousePosX = x;
		input.mousePosY = y;
		break;
	}
	case WM_MOUSEWHEEL:
	{
		int delta = GET_WHEEL_DELTA_WPARAM(wParam);
		input.mouseDelta = delta;
		break;
	}
	case WM_PAINT:
	{
		PAINTSTRUCT paint;
		HDC deviceContext = BeginPaint(hwnd, &paint);
		int x = paint.rcPaint.left;
		int y = paint.rcPaint.top;
		int width = paint.rcPaint.right - paint.rcPaint.left;
		int height = paint.rcPaint.bottom - paint.rcPaint.top;
		Win32UpdateWindow(deviceContext, x, y, width, height);
		EndPaint(hwnd, &paint);
		break;
	}
	default:
	{
		result = DefWindowProc(hwnd, uMsg, wParam, lParam);
		break;
	}
	}

	return result;
}

void CreateConsole()
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);

	if (AllocConsole()){
		freopen("CONIN$", "r", stdin);
		freopen("CONOUT$", "w", stdout);
		freopen("CONOUT$", "w", stderr);
	}
}

void StartTime()
{
	frameCount = 0;
	QueryPerformanceCounter(&lastCounter);
	QueryPerformanceFrequency(&perfCounterFrequencyResult);
	lastCycleCount = __rdtsc();
	perfCounterFrequency = perfCounterFrequencyResult.QuadPart;

	time.frameCount = 0;
	time.time = 0.0f;
	time.deltaTime = 0.0f;
	time.deltaCycles = 0;
}

void RecordTime()
{
	unsigned long long endCycleCount = __rdtsc();
	LARGE_INTEGER endCounter;
	QueryPerformanceCounter(&endCounter);

	unsigned long long cyclesElapsed = endCycleCount - lastCycleCount;
	long long counterElapsed = endCounter.QuadPart - lastCounter.QuadPart;
	float megaCyclesPerFrame = ((float)cyclesElapsed / (1000.0f * 1000.0f));
	float milliSecPerFrame = ((1000.0f*(float)counterElapsed) / (float)perfCounterFrequency);

	lastCycleCount = endCycleCount;
	lastCounter = endCounter;
	frameCount++;

	time.frameCount = frameCount;
	time.time = (1000.0f * (float)lastCounter.QuadPart) / ((float)perfCounterFrequency);
	time.deltaTime = ((1000.0f*(float)counterElapsed) / (float)perfCounterFrequency);
	time.deltaCycles = (size_t)cyclesElapsed;
}


int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	CreateConsole();

	WNDCLASS windowClass = {};
	windowClass.style = CS_HREDRAW | CS_VREDRAW;
	windowClass.lpfnWndProc = MainWindowCallback;
	windowClass.hInstance = hInstance;
	windowClass.lpszClassName = _T("PhysicsEngineWindowClass");

	if (!RegisterClass(&windowClass)){
		return 0;
	}

	HWND windowHandle = CreateWindowEx(0, 
		windowClass.lpszClassName, 
		_T("PhysicsEngine"), 
		WS_OVERLAPPEDWINDOW | WS_VISIBLE, 
		CW_USEDEFAULT, 
		CW_USEDEFAULT, 
		1024, 
		1024, 
		0, 
		0, 
		hInstance, 
		0);

	if (!windowHandle){
		return 0;
	}

	if (!Win32InitOpenGL(windowHandle)){  
		return 0;
	}

	Scene scene;
	AssetBundle assetBundle;

	scene.filepath = "C:\\Users\\James\\Documents\\PhysicsEngine\\sample_project\\Demo\\x64\\Release\\drawcall.scene";
	assetBundle.filepath = "C:\\Users\\James\\Documents\\PhysicsEngine\\sample_project\\Demo\\x64\\Release\\bundle.assets";

	WorldManager worldManager(scene, assetBundle);

	worldManager.init();

	StartTime();

	running = true;

	while (running)
	{
		MSG message;
		while (PeekMessage(&message, 0, 0, 0, PM_REMOVE))
		{
			if (message.message == WM_QUIT){ running = false; }

			TranslateMessage(&message);
			DispatchMessage(&message);
		}

		for (DWORD controllerIndex = 0; controllerIndex < XUSER_MAX_COUNT; controllerIndex++)
		{
			XINPUT_STATE ControllerState;
			if (XInputGetState(controllerIndex, &ControllerState) == ERROR_SUCCESS){
				XINPUT_GAMEPAD *Pad = &ControllerState.Gamepad;

				input.xboxButtonIsDown[(int)XboxButton::LeftDPad] = (Pad->wButtons & XINPUT_GAMEPAD_DPAD_LEFT);
				input.xboxButtonIsDown[(int)XboxButton::RightDPad] = (Pad->wButtons & XINPUT_GAMEPAD_DPAD_RIGHT);
				input.xboxButtonIsDown[(int)XboxButton::UpDPad] = (Pad->wButtons & XINPUT_GAMEPAD_DPAD_UP);
				input.xboxButtonIsDown[(int)XboxButton::DownDPad] = (Pad->wButtons & XINPUT_GAMEPAD_DPAD_DOWN);
				input.xboxButtonIsDown[(int)XboxButton::Start] = (Pad->wButtons & XINPUT_GAMEPAD_START);
				input.xboxButtonIsDown[(int)XboxButton::Back] = (Pad->wButtons & XINPUT_GAMEPAD_BACK);
				input.xboxButtonIsDown[(int)XboxButton::LeftThumb] = (Pad->wButtons & XINPUT_GAMEPAD_LEFT_THUMB);
				input.xboxButtonIsDown[(int)XboxButton::RightThumb] = (Pad->wButtons & XINPUT_GAMEPAD_RIGHT_THUMB);
				input.xboxButtonIsDown[(int)XboxButton::LeftShoulder] = (Pad->wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER);
				input.xboxButtonIsDown[(int)XboxButton::RightShoulder] = (Pad->wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER);
				input.xboxButtonIsDown[(int)XboxButton::AButton] = (Pad->wButtons & XINPUT_GAMEPAD_A);
				input.xboxButtonIsDown[(int)XboxButton::BButton] = (Pad->wButtons & XINPUT_GAMEPAD_B);
				input.xboxButtonIsDown[(int)XboxButton::XButton] = (Pad->wButtons & XINPUT_GAMEPAD_X);
				input.xboxButtonIsDown[(int)XboxButton::YButton] = (Pad->wButtons & XINPUT_GAMEPAD_Y);

				input.leftStickX = Pad->sThumbLX;
				input.leftStickY = Pad->sThumbLY;
				input.rightStickX = Pad->sThumbRX;
				input.rightStickY = Pad->sThumbRY;
			}
			else{

			}
		}

		worldManager.update(time, input);

		RedrawWindow(windowHandle, 0, 0, RDW_INVALIDATE);

		RecordTime();

		// input
		for(int i = 0; i < 3; i++){
			input.mouseButtonWasDown[i] = input.mouseButtonIsDown[i];
			//input.buttonIsDown[i] = false;
		}

		input.mouseDelta = 0;

		for(int i = 0; i < 51; i++){
		 	input.keyIsDown[i] = false;
			input.keyWasDown[i] = false;// input.keyIsDown[i];// false
		}

		for (int i = 0; i < 14; i++){
			input.xboxButtonWasDown[i] = input.xboxButtonIsDown[i];
			input.xboxButtonIsDown[i] = false;
		}
	}

	return 0;
}
















//#define IS_INTRESOURCE(_r) ((((ULONG_PTR)(_r)) >> 16) == 0)
//#define MAKEINTRESOURCEA(i) ((LPSTR)((ULONG_PTR)((WORD)(i))))
//#define MAKEINTRESOURCEW(i) ((LPWSTR)((ULONG_PTR)((WORD)(i))))
//#ifdef UNICODE
//#define MAKEINTRESOURCE  MAKEINTRESOURCEW
//#else
//#define MAKEINTRESOURCE  MAKEINTRESOURCEA
//#endif // !UNICODE
//
//#define CreateWindowA(lpClassName, lpWindowName, dwStyle, x, y,\
//nWidth, nHeight, hWndParent, hMenu, hInstance, lpParam)\
//CreateWindowExA(0L, lpClassName, lpWindowName, dwStyle, x, y,\
//nWidth, nHeight, hWndParent, hMenu, hInstance, lpParam)
//#define CreateWindowW(lpClassName, lpWindowName, dwStyle, x, y,\
//nWidth, nHeight, hWndParent, hMenu, hInstance, lpParam)\
//CreateWindowExW(0L, lpClassName, lpWindowName, dwStyle, x, y,\
//nWidth, nHeight, hWndParent, hMenu, hInstance, lpParam)
//#ifdef UNICODE
//#define CreateWindow  CreateWindowW
//#else
//#define CreateWindow  CreateWindowA
//#endif // !UNICODE
//
//#ifdef UNICODE
//#define MAKEINTRESOURCE  MAKEINTRESOURCEW
//#else
//#define MAKEINTRESOURCE  MAKEINTRESOURCEA
//#endif // !UNICODE
//
//#define IDI_ICON1                                4001
//
//#define GAME_NAME						"DOOM 3"	
//
//#define	COMMAND_HISTORY	64
//#define GWL_WNDPROC         (-4)
//#define INPUT_ID		101
//#define EDIT_ID			100
//
//typedef struct {
//	HWND			hWnd;
//	HINSTANCE		hInstance;
//
//	OSVERSIONINFOEX	osversion;
//
//	//cpuid_t			cpuid;
//
//	// when we get a windows message, we store the time off so keyboard processing
//	// can know the exact time of an event (not really needed now that we use async direct input)
//	int				sysMsgTime;
//
//	bool			windowClassRegistered;
//
//	WNDPROC			wndproc;
//
//	HDC				hDC;							// handle to device context
//	HGLRC			hGLRC;						// handle to GL rendering context
//	PIXELFORMATDESCRIPTOR pfd;
//	int				pixelformat;
//
//	HINSTANCE		hinstOpenGL;	// HINSTANCE for the OpenGL library
//
//	int				desktopBitsPixel;
//	int				desktopWidth, desktopHeight;
//
//	bool			cdsFullscreen;
//
//	FILE			*log_fp;
//
//} Win32Vars_t;
//
//typedef struct {
//	HWND		hWnd;
//	HWND		hwndBuffer;
//
//	HWND		hwndButtonClear;
//	HWND		hwndButtonCopy;
//	HWND		hwndButtonQuit;
//
//	HWND		hwndErrorBox;
//	HWND		hwndErrorText;
//
//	HBITMAP		hbmLogo;
//	HBITMAP		hbmClearBitmap;
//
//	HBRUSH		hbrEditBackground;
//	HBRUSH		hbrErrorBackground;
//
//	HFONT		hfBufferFont;
//	HFONT		hfButtonFont;
//
//	HWND		hwndInputLine;
//
//	char		errorString[80];
//
//	char		consoleText[512], returnedText[512];
//	bool		quitOnClose;
//	int			windowWidth, windowHeight;
//
//	WNDPROC		SysInputLineWndProc;
//
//	//idEditField	historyEditLines[COMMAND_HISTORY];
//
//	int			nextHistoryLine;// the last line in the history buffer, not masked
//	int			historyLine;	// the line being displayed from history buffer
//
//} WinConData;
//
//
//
//Win32Vars_t	win32;
//static WinConData s_wcd;
//
//static LONG WINAPI ConWndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
//	return 0;
//}
//
//LONG WINAPI InputLineWndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
//	return 0;
//}
//
///*
//** Sys_CreateConsole
//*/
//void Sys_CreateConsole(void) {
//	HDC hDC;
//	WNDCLASS wc;
//	RECT rect;
//	const char *DEDCLASS = "DOOM 3 WinConsole";// WIN32_CONSOLE_CLASS;
//	int nHeight;
//	int swidth, sheight;
//	int DEDSTYLE = WS_POPUPWINDOW | WS_CAPTION | WS_MINIMIZEBOX;
//	int i;
//
//	memset(&wc, 0, sizeof(wc));
//
//	wc.style = 0;
//	wc.lpfnWndProc = (WNDPROC)MainWindowCallback;
//	wc.cbClsExtra = 0;
//	wc.cbWndExtra = 0;
//	wc.hInstance = win32.hInstance;
//	wc.hIcon = LoadIcon(win32.hInstance, MAKEINTRESOURCE(IDI_ICON1));
//	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
//	wc.hbrBackground = (struct HBRUSH__ *)COLOR_WINDOW;
//	wc.lpszMenuName = 0;
//	wc.lpszClassName = DEDCLASS;
//
//	if (!RegisterClass(&wc)) {
//		return;
//	}
//
//	rect.left = 0;
//	rect.right = 540;
//	rect.top = 0;
//	rect.bottom = 450;
//	AdjustWindowRect(&rect, DEDSTYLE, FALSE);
//
//	hDC = GetDC(GetDesktopWindow());
//	swidth = GetDeviceCaps(hDC, HORZRES);
//	sheight = GetDeviceCaps(hDC, VERTRES);
//	ReleaseDC(GetDesktopWindow(), hDC);
//
//	s_wcd.windowWidth = rect.right - rect.left + 1;
//	s_wcd.windowHeight = rect.bottom - rect.top + 1;
//
//	//s_wcd.hbmLogo = LoadBitmap( win32.hInstance, MAKEINTRESOURCE( IDB_BITMAP_LOGO) );
//
//	s_wcd.hWnd = CreateWindowEx(0,
//		DEDCLASS,
//		GAME_NAME,
//		DEDSTYLE,
//		(swidth - 600) / 2, (sheight - 450) / 2, rect.right - rect.left + 1, rect.bottom - rect.top + 1,
//		NULL,
//		NULL,
//		win32.hInstance,
//		NULL);
//
//	if (s_wcd.hWnd == NULL) {
//		return;
//	}
//
//	////
//	//// create fonts
//	////
//	//hDC = GetDC(s_wcd.hWnd);
//	//nHeight = -MulDiv(8, GetDeviceCaps(hDC, LOGPIXELSY), 72);
//
//	//s_wcd.hfBufferFont = CreateFont(nHeight, 0, 0, 0, FW_LIGHT, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, FF_MODERN | FIXED_PITCH, "Courier New");
//
//	//ReleaseDC(s_wcd.hWnd, hDC);
//
//	////
//	//// create the input line
//	////
//	//s_wcd.hwndInputLine = CreateWindow("edit", NULL, WS_CHILD | WS_VISIBLE | WS_BORDER |
//	//	ES_LEFT | ES_AUTOHSCROLL,
//	//	6, 400, 528, 20,
//	//	s_wcd.hWnd,
//	//	(HMENU)INPUT_ID,	// child window ID
//	//	win32.hInstance, NULL);
//
//	////
//	//// create the scrollbuffer
//	////
//	//s_wcd.hwndBuffer = CreateWindow("edit", NULL, WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_BORDER |
//	//	ES_LEFT | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
//	//	6, 40, 526, 354,
//	//	s_wcd.hWnd,
//	//	(HMENU)EDIT_ID,	// child window ID
//	//	win32.hInstance, NULL);
//	//SendMessage(s_wcd.hwndBuffer, WM_SETFONT, (WPARAM)s_wcd.hfBufferFont, 0);
//
//	//s_wcd.SysInputLineWndProc = (WNDPROC)SetWindowLong(s_wcd.hwndInputLine, GWL_WNDPROC, (long)InputLineWndProc);
//	//SendMessage(s_wcd.hwndInputLine, WM_SETFONT, (WPARAM)s_wcd.hfBufferFont, 0);
//}
//
///*
//** Sys_DestroyConsole
//*/
//void Sys_DestroyConsole(void) {
//	if (s_wcd.hWnd) {
//		ShowWindow(s_wcd.hWnd, SW_HIDE);
//		CloseWindow(s_wcd.hWnd);
//		DestroyWindow(s_wcd.hWnd);
//		s_wcd.hWnd = 0;
//	}
//}
//
///*
//** Sys_ShowConsole
//*/
//void Sys_ShowConsole(int visLevel, bool quitOnClose) {
//
//	s_wcd.quitOnClose = quitOnClose;
//
//	if (!s_wcd.hWnd) {
//		return;
//	}
//
//	switch (visLevel) {
//	case 0:
//		ShowWindow(s_wcd.hWnd, SW_HIDE);
//		break;
//	case 1:
//		ShowWindow(s_wcd.hWnd, SW_SHOWNORMAL);
//		SendMessage(s_wcd.hwndBuffer, EM_LINESCROLL, 0, 0xffff);
//		break;
//	case 2:
//		ShowWindow(s_wcd.hWnd, SW_MINIMIZE);
//		break;
//	default:
//		//Sys_Error("Invalid visLevel %d sent to Sys_ShowConsole\n", visLevel);
//		break;
//	}
//}