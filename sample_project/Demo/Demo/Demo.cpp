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

Input input;

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
		input.buttonIsDown[(int)LButton] = true;
		input.buttonWasDown[(int)LButton] = false;
		break;
	}
	case WM_MBUTTONDOWN:
	{
		input.buttonIsDown[(int)MButton] = true;
		input.buttonWasDown[(int)MButton] = false;
		break;
	}
	case WM_RBUTTONDOWN:
	{
		input.buttonIsDown[(int)RButton] = true;
		input.buttonWasDown[(int)RButton] = false;
		break;
	}
	case WM_LBUTTONUP:
	{
		input.buttonIsDown[(int)LButton] = false;
		input.buttonWasDown[(int)LButton] = true;
		break;
	}
	case WM_MBUTTONUP:
	{
		input.buttonIsDown[(int)MButton] = false;
		input.buttonWasDown[(int)MButton] = true;
		break;
	}
	case WM_RBUTTONUP:
	{
		input.buttonIsDown[(int)RButton] = false;
		input.buttonWasDown[(int)RButton] = true;
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

int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);

	if (AllocConsole()){
		freopen("CONIN$", "r", stdin);
		freopen("CONOUT$", "w", stdout);
		freopen("CONOUT$", "w", stderr);
	}
	else{
		std::cout << "Error: Failed to allocate console" << std::endl;
		return 0;
	}

	WNDCLASS windowClass = {};
	windowClass.style = CS_HREDRAW | CS_VREDRAW;
	windowClass.lpfnWndProc = MainWindowCallback;
	windowClass.hInstance = hInstance;
	windowClass.lpszClassName = _T("PhysicsEngineWindowClass");

	if (!RegisterClass(&windowClass)){
		std::cout << "Error: Failed to register window class" << std::endl;
		return 0;
	}

	HWND windowHandle = CreateWindowEx(0, windowClass.lpszClassName, _T("PhysicsEngine"), WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, 1000, 1000, 0, 0, hInstance, 0);

	if (!windowHandle){
		std::cout << "Error: Failed tocreate window handle" << std::endl;
		return 0;
	}

	if (!Win32InitOpenGL(windowHandle)){  //call something like Graphics::init()??
		std::cout << "Error: Failed to initialize graphics API" << std::endl;
		return 0;
	}

	Scene scene;
	AssetBundle assetBundle;

	scene.filepath = "C:\\Users\\James\\Documents\\PhysicsEngine\\sample_project\\Demo\\x64\\Release\\simple.scene";
	assetBundle.filepath = "C:\\Users\\James\\Documents\\PhysicsEngine\\sample_project\\Demo\\x64\\Release\\bundle.assets";

	WorldManager worldManager(scene, assetBundle);

	worldManager.init();

	// total frame timing
	int frameCount = 0;
	LARGE_INTEGER lastCounter;
	LARGE_INTEGER perfCounterFrequencyResult;
	QueryPerformanceCounter(&lastCounter);
	QueryPerformanceFrequency(&perfCounterFrequencyResult);
	unsigned long long lastCycleCount = __rdtsc();
	long long perfCounterFrequency = perfCounterFrequencyResult.QuadPart;

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

		if (!worldManager.update(input)){ running = false; }

		RedrawWindow(windowHandle, 0, 0, RDW_INVALIDATE);

		// record time
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

		Time::frameCount = frameCount;
		Time::deltaCycles = (int)cyclesElapsed;
		Time::time = (1000.0f * (float)lastCounter.QuadPart) / ((float)perfCounterFrequency);
		Time::deltaTime = milliSecPerFrame;

		// input
		for(int i = 0; i < 3; i++){
			input.buttonWasDown[i] = input.buttonIsDown[i];
			//input.buttonIsDown[i] = false;
		}

		input.mouseDelta = 0;

		for(int i = 0; i < 51; i++){
		 	input.keyIsDown[i] = false;
			input.keyWasDown[i] = false;// input.keyIsDown[i];// false
		}
	}

	return 0;
}