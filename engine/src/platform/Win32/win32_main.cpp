#include <windows.h>
#include <windowsx.h>
#include <xinput.h>
#include <GL/glew.h>
#include <gl/gl.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

#include <cuda.h>
#include <cudagl.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifndef WGL_EXT_extensions_string
#define WGL_EXT_extensions_string 1
#ifdef WGL_WGLEXT_PROTOTYPES
extern const char * WINAPI wglGetExtensionsStringEXT (void);
#endif /* WGL_WGLEXT_PROTOTYPES */
typedef const char * (WINAPI * PFNWGLGETEXTENSIONSSTRINGEXTPROC) (void);
#endif

#ifndef WGL_EXT_swap_control
#define WGL_EXT_swap_control 1
#ifdef WGL_WGLEXT_PROTOTYPES
extern BOOL WINAPI wglSwapIntervalEXT (int);
extern int WINAPI wglGetSwapIntervalEXT (void);
#endif /* WGL_WGLEXT_PROTOTYPES */
typedef BOOL (WINAPI * PFNWGLSWAPINTERVALEXTPROC) (int interval);
typedef int (WINAPI * PFNWGLGETSWAPINTERVALEXTPROC) (void);
#endif

#include "../../../include/cuda/cuda_util.h"

#include "../../../include/core/Input.h"
#include "../../../include/core/Time.h"
#include "../../../include/core/Log.h"
#include "../../../include/core/SceneManager.h"

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

std::vector<std::string> get_all_asset_files(std::string relativePath)
{
	std::vector<std::string> materialFilePaths = get_all_files_names_within_folder(relativePath + "materials/", "mat");
	std::vector<std::string> meshFilePaths = get_all_files_names_within_folder(relativePath + "meshes/", "mesh");
	std::vector<std::string> gmeshFilePaths = get_all_files_names_within_folder(relativePath + "gmeshes/", "gmesh");
	std::vector<std::string> textureFilePaths = get_all_files_names_within_folder(relativePath + "textures/", "png");
	std::vector<std::string> shaderFilePaths = get_all_files_names_within_folder(relativePath + "shaders/", "shader");

	std::vector<std::string> assetFilePaths;
	for(unsigned int i = 0; i < materialFilePaths.size(); i++){ assetFilePaths.push_back(materialFilePaths[i]); }
	for(unsigned int i = 0; i < meshFilePaths.size(); i++){ assetFilePaths.push_back(meshFilePaths[i]); }
	for(unsigned int i = 0; i < gmeshFilePaths.size(); i++){ assetFilePaths.push_back(gmeshFilePaths[i]); }
	for(unsigned int i = 0; i < textureFilePaths.size(); i++){ assetFilePaths.push_back(textureFilePaths[i]); }
	for(unsigned int i = 0; i < shaderFilePaths.size(); i++){ assetFilePaths.push_back(shaderFilePaths[i]); }

	return assetFilePaths;
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

bool WGLExtensionSupported(const char *extension_name)
{
    // this is pointer to function which returns pointer to string with list of all wgl extensions
    PFNWGLGETEXTENSIONSSTRINGEXTPROC _wglGetExtensionsStringEXT = NULL;

    // determine pointer to wglGetExtensionsStringEXT function
    _wglGetExtensionsStringEXT = (PFNWGLGETEXTENSIONSSTRINGEXTPROC) wglGetProcAddress("wglGetExtensionsStringEXT");

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

	PFNWGLSWAPINTERVALEXTPROC       wglSwapIntervalEXT = NULL;
	PFNWGLGETSWAPINTERVALEXTPROC    wglGetSwapIntervalEXT = NULL;

	if (WGLExtensionSupported("WGL_EXT_swap_control"))
	{
		std::cout << "vsync extension supported" << std::endl;

		// Extension is supported, init pointers.
	    wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC) wglGetProcAddress("wglSwapIntervalEXT");

	    // this is another function from WGL_EXT_swap_control extension
	    wglGetSwapIntervalEXT = (PFNWGLGETSWAPINTERVALEXTPROC) wglGetProcAddress("wglGetSwapIntervalEXT");

	    wglSwapIntervalEXT(0);//vsync On: 1 Off: 0
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

		HWND windowHandle = CreateWindowEx(0, windowClass.lpszClassName, "PhysicsEngine", WS_OVERLAPPEDWINDOW|WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, 1000, 1000, 0, 0, hInstance, 0);

		if(windowHandle)
		{
			if(!Win32InitOpenGL(windowHandle)){ return 0; }

			SceneManager sceneManager;

			// fill in scene manager with all scenes found in the scene build register
			std::string sceneFileName;
		  	std::ifstream sceneRegisterFile ("../data/scene_build_register.txt");
		  	if (sceneRegisterFile.is_open()){
		   		while ( getline (sceneRegisterFile, sceneFileName) ){
		   			Scene scene;
		   			scene.name = sceneFileName;
		   			scene.filepath = "../data/scenes/" + sceneFileName;
		   			scene.isLoaded = false;

		   			sceneManager.add(scene);

		      		std::cout << sceneFileName << std::endl;
		    	}

		    	sceneRegisterFile.close();
		  	}
		  	else{
		  		std::cout << "Error: Could not open scene build register file" << std::endl;
		  	}

		  	// fill in scene manager with all assets found in the data folder
			std::vector<std::string> assetFilePaths = get_all_asset_files("../data/");

			for(unsigned int i = 0; i < assetFilePaths.size(); i++){ 
				AssetFile assetFile;
				assetFile.filepath = assetFilePaths[i];

				sceneManager.add(assetFile); 
			}

			if(sceneManager.validate()){
				sceneManager.init();

				running = true;

				// total frame timing
				int frameCount = 0;
				LARGE_INTEGER lastCounter;
				QueryPerformanceCounter(&lastCounter);
				unsigned long long lastCycleCount = __rdtsc();

				// gpu only timing
				GLuint query;
				GLuint gpu_time;
				glGenQueries(1, &query);

				while(running)
				{
					glBeginQuery(GL_TIME_ELAPSED, query);

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

					sceneManager.update();

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

					glEndQuery(GL_TIME_ELAPSED);
					glGetQueryObjectuiv(query, GL_QUERY_RESULT, &gpu_time);

					Time::frameCount = frameCount;
					Time::deltaCycles = (int)cyclesElapsed;
					Time::time = (1000.0f * (float)lastCounter.QuadPart) / ((float)perfCounterFrequency);
					Time::deltaTime = milliSecPerFrame;
					Time::gpuDeltaTime = gpu_time / 1000000.0f;

					Input::updateEOF();
				}
			}
			else{
				while(true){

				}
			}
		}
		else{
			// TODO handle unlikely error?
		}
	}
	else{
		// TODO handle unlikely error?
	}
	
	return 0;
}