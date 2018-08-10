#include <windows.h>
#include <windowsx.h>
#include <xinput.h>
#include <GL/glew.h>
#include <gl/gl.h>
#include <stdio.h>

#include <cuda.h>
#include <cudagl.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "../../../include/cuda/cuda_util.h"

#include "../../../include/core/Input.h"
#include "../../../include/core/Time.h"
#include "../../../include/core/Log.h"
#include "../../../include/core/Scene.h"

using namespace PhysicsEngine;

static bool running;

std::vector<std::string> get_all_files_names_within_folder(std::string folder, std::string extension)
{
    std::vector<std::string> names;
    std::string search_path = folder + "/*.*";
    WIN32_FIND_DATA fd; 
    HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd); 
    if(hFind != INVALID_HANDLE_VALUE) { 
        do { 
            // read all (real) files in current folder
            // , delete '!' read other 2 default folder . and ..
            if(! (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ) {

            	std::string file = fd.cFileName;
            	if(file.substr(file.find_last_of(".") + 1) == extension) {
            		names.push_back(folder + file);
				}
                //names.push_back(fd.cFileName);
            }
        }while(::FindNextFile(hFind, &fd)); 
        ::FindClose(hFind); 
    } 
    return names;
}

KeyCode GetKeyCode(unsigned int vKCode)
{
	KeyCode keyCode;
	switch(vKCode)
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
		default:{ keyCode = KeyCode::Invalid; break; }
	}	

	return keyCode;
}


bool Win32InitOpenGL(HWND window)
{
	HDC windowDC = GetDC(window);

	PIXELFORMATDESCRIPTOR desiredPixelFormat = {};
	desiredPixelFormat.nSize = sizeof(desiredPixelFormat);
	desiredPixelFormat.nVersion = 1;
	desiredPixelFormat.dwFlags = PFD_SUPPORT_OPENGL|PFD_DRAW_TO_WINDOW|PFD_DOUBLEBUFFER;
	desiredPixelFormat.cColorBits = 32;
	desiredPixelFormat.cAlphaBits = 8;
	desiredPixelFormat.iLayerType = PFD_MAIN_PLANE;

	int suggestedPixelFormatIndex = ChoosePixelFormat(windowDC, &desiredPixelFormat);

	PIXELFORMATDESCRIPTOR suggestedPixelFormat;
	DescribePixelFormat(windowDC, suggestedPixelFormatIndex, sizeof(suggestedPixelFormat), &suggestedPixelFormat);
	SetPixelFormat(windowDC, suggestedPixelFormatIndex, &suggestedPixelFormat);
	
	HGLRC openGLRC = wglCreateContext(windowDC);
	//wglCreateContextAttrib();
	if(!wglMakeCurrent(windowDC, openGLRC))
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
		sprintf(buffer, "glew error code %d\n", err);
		OutputDebugStringA(buffer);
		return false;
	}
	else
	{
		OutputDebugStringA("INITIALIZED GLEW SUCCESSFULLY\n");
	}

	cudaDeviceProp prop;
	int device; 
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	gpuErrchk(cudaChooseDevice(&device, &prop));
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

	switch(uMsg)
	{
		case WM_SIZE:
		{
			OutputDebugStringA("WM_SIZE CALLED\n");
			break;
		}
		case WM_CLOSE:
			running = false;
			PostQuitMessage(0);
			break;
		case WM_DESTROY:
		{
			running = false;
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
			Input::setKeyState(keyCode, isDown, wasDown);
			break;
		}
		case WM_LBUTTONDOWN:
		{
			Input::setMouseButtonState(LButton, true, false);
			break;
		}
		case WM_MBUTTONDOWN:
		{
			Input::setMouseButtonState(MButton, true, false);
			break;
		}
		case WM_RBUTTONDOWN:
		{
			Input::setMouseButtonState(RButton, true, false);
			break;
		}
		case WM_LBUTTONUP:
		{
			Input::setMouseButtonState(LButton, false, true);
			break;
		}
		case WM_MBUTTONUP:
		{
			Input::setMouseButtonState(MButton, false, true);
			break;
		}
		case WM_RBUTTONUP:
		{
			Input::setMouseButtonState(RButton, false, true);
			break;
		}
		case WM_MOUSEMOVE:
		{
			int x = GET_X_LPARAM(lParam);
			int y = GET_Y_LPARAM(lParam);
			Input::setMousePosition(x, y);
			break;
		}
		case WM_MOUSEWHEEL:
		{
			int delta = GET_WHEEL_DELTA_WPARAM(wParam);

			Input::setMouseDelta(delta);
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
	MessageBox(0, GetCommandLine(), "PhysicsEngine", MB_OK|MB_ICONINFORMATION);
	MessageBox(0, lpCmdLine, "PhysicsEngine", MB_OK|MB_ICONINFORMATION);
	
	// std::cout << "commanad line: " << lpCmdLine << std::endl;

	LARGE_INTEGER perfCounterFrequencyResult;
	QueryPerformanceFrequency(&perfCounterFrequencyResult);
	long long perfCounterFrequency = perfCounterFrequencyResult.QuadPart;

	WNDCLASS windowClass = {};
	windowClass.style = CS_HREDRAW|CS_VREDRAW;
	windowClass.lpfnWndProc = MainWindowCallback;
	windowClass.hInstance = hInstance;
	windowClass.lpszClassName = "PhysicsEngineWindowClass";
	
	if(RegisterClass(&windowClass))
	{
		AllocConsole();
		freopen("CONIN$", "r",stdin);
		freopen("CONOUT$", "w",stdout);
		freopen("CONOUT$", "w",stderr);

		HWND windowHandle = CreateWindowEx(0, 
				windowClass.lpszClassName, 
				"PhysicsEngine", 
				WS_OVERLAPPEDWINDOW|WS_VISIBLE,
				CW_USEDEFAULT,
				CW_USEDEFAULT,
				1000,
				1000,
				0,
				0,
				hInstance,
				0);
		if(windowHandle)
		{
			if(!Win32InitOpenGL(windowHandle)){ return 0; }

			// init game?
			Scene scene;

			// load assets
			std::vector<std::string> materialFilePaths = get_all_files_names_within_folder("../data/materials/", "mat");
			std::vector<std::string> meshFilePaths = get_all_files_names_within_folder("../data/meshes/", "mesh");
			std::vector<std::string> gmeshFilePaths = get_all_files_names_within_folder("../data/gmeshes/", "gmesh");
			std::vector<std::string> textureFilePaths = get_all_files_names_within_folder("../data/textures/", "png");
			std::vector<std::string> shaderFilePaths = get_all_files_names_within_folder("../data/shaders/", "shader");

			std::vector<std::string> assetFilePaths;
			for(unsigned int i = 0; i < materialFilePaths.size(); i++){ assetFilePaths.push_back(materialFilePaths[i]); }
			for(unsigned int i = 0; i < meshFilePaths.size(); i++){ assetFilePaths.push_back(meshFilePaths[i]); }
			for(unsigned int i = 0; i < gmeshFilePaths.size(); i++){ assetFilePaths.push_back(gmeshFilePaths[i]); }
			for(unsigned int i = 0; i < textureFilePaths.size(); i++){ assetFilePaths.push_back(textureFilePaths[i]); }
			for(unsigned int i = 0; i < shaderFilePaths.size(); i++){ assetFilePaths.push_back(shaderFilePaths[i]); }	

			if(scene.validate("../data/scenes/simple.scene", assetFilePaths)){
				std::cout << "Calling scene load" << std::endl;
				// scene.load(lpCmdLine, assetFilePaths);
				scene.load("../data/scenes/simple.scene", assetFilePaths);

				running = true;

				int frameCount = 0;
				LARGE_INTEGER lastCounter;
				QueryPerformanceCounter(&lastCounter);
				unsigned long long lastCycleCount = __rdtsc();
				while(running)
				{
					MSG message;
					while(PeekMessage(&message, 0, 0, 0, PM_REMOVE))
					{
						if(message.message == WM_QUIT)
						{
							running = false;
						}

						TranslateMessage(&message);
						DispatchMessage(&message);
					}

					// controller input
					for(DWORD controllerIndex = 0; controllerIndex < XUSER_MAX_COUNT; controllerIndex++){
						XINPUT_STATE controllerState;
						if(XInputGetState(controllerIndex, &controllerState) == ERROR_SUCCESS){
							XINPUT_GAMEPAD *pad = &controllerState.Gamepad;
							// bool up = (pad->wButtons & XINPUT_GAMEPAD_DPAD_UP);
							// bool down = (pad->wButtons & XINPUT_GAMEPAD_DPAD_DOWN);
							// bool left = (pad->wButtons & XINPUT_GAMEPAD_DPAD_LEFT);
							// bool right = (pad->wButtons & XINPUT_GAMEPAD_DPAD_RIGHT);
							// bool start = (pad->wButtons & XINPUT_GAMEPAD_START);
							// bool back = (pad->wButtons & XINPUT_GAMEPAD_BACK);
							// bool leftShoulder = (pad->wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER);
							// bool rightShoulder = (pad->wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER);
							// bool aButton = (pad->wButtons & XINPUT_GAMEPAD_A);
							// bool bButton = (pad->wButtons & XINPUT_GAMEPAD_B);
							// bool xButton = (pad->wButtons & XINPUT_GAMEPAD_X);
							// bool yButton = (pad->wButtons & XINPUT_GAMEPAD_Y);
						}
						else{
							// NOTE: controller not available
						}

					}

					// run game update?
					scene.update();

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

					// char buffer[256];
					// sprintf(buffer, "frame count %d delta cycles %d time %f delta time %f\n", Time::frameCount, Time::deltaCycles, Time::time, Time::deltaTime);
					// OutputDebugStringA(buffer);

					Input::updateEOF();
				}
			}
			else
			{
				std::cout << "Failed scene validation" << std::endl;
			}
		}
		else
		{
			// TODO handle unlikely error?
		}
	}
	else
	{
		// TODO handle unlikely error?
	}

	return 0;
}