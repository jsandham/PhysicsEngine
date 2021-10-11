#include "../../../include/core/platform/RendererAPI_win32_opengl.h"

#include <stdio.h>
#include <GL/glew.h>

using namespace PhysicsEngine;

RendererAPI_win32_opengl::RendererAPI_win32_opengl()
{    
}

RendererAPI_win32_opengl::~RendererAPI_win32_opengl()
{
}

void RendererAPI_win32_opengl::init(void* window)
{
    // Prepare OpenGlContext
    CreateGlContext(window);
    SetSwapInterval(1);

    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    }
}

void RendererAPI_win32_opengl::update()
{
    SwapBuffers(g_HDCDeviceContext);
}

void RendererAPI_win32_opengl::cleanup()
{
    wglDeleteContext(g_GLRenderContext);
}

void RendererAPI_win32_opengl::CreateGlContext(void* window)
{
    PIXELFORMATDESCRIPTOR desiredPixelFormat = {};
    desiredPixelFormat.nSize = sizeof(desiredPixelFormat);
    desiredPixelFormat.nVersion = 1;
    desiredPixelFormat.dwFlags = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER; // PFD_DRAW_TO_BITMAP
    desiredPixelFormat.cColorBits = 32;
    desiredPixelFormat.cAlphaBits = 8;
    desiredPixelFormat.iLayerType = PFD_MAIN_PLANE;

    g_HDCDeviceContext = GetDC(static_cast<HWND>(window));

    int suggestedPixelFormatIndex = ChoosePixelFormat(g_HDCDeviceContext, &desiredPixelFormat);

    PIXELFORMATDESCRIPTOR suggestedPixelFormat;
    DescribePixelFormat(g_HDCDeviceContext, suggestedPixelFormatIndex, sizeof(suggestedPixelFormat),
        &suggestedPixelFormat);
    SetPixelFormat(g_HDCDeviceContext, suggestedPixelFormatIndex, &suggestedPixelFormat);

    g_GLRenderContext = wglCreateContext(g_HDCDeviceContext);
    if (!wglMakeCurrent(g_HDCDeviceContext, g_GLRenderContext))
    {
        return;
    }
}

#ifndef WGL_EXT_extensions_string
#define WGL_EXT_extensions_string 1
#ifdef WGL_WGLEXT_PROTOTYPES
extern const char* WINAPI wglGetExtensionsStringEXT(void);
#endif /* WGL_WGLEXT_PROTOTYPES */
typedef const char* (WINAPI* PFNWGLGETEXTENSIONSSTRINGEXTPROC)(void);
#endif

#ifndef WGL_EXT_swap_control
#define WGL_EXT_swap_control 1
#ifdef WGL_WGLEXT_PROTOTYPES
extern BOOL WINAPI wglSwapIntervalEXT(int);
extern int WINAPI wglGetSwapIntervalEXT(void);
#endif /* WGL_WGLEXT_PROTOTYPES */
typedef BOOL(WINAPI* PFNWGLSWAPINTERVALEXTPROC)(int interval);
typedef int(WINAPI* PFNWGLGETSWAPINTERVALEXTPROC)(void);
#endif

bool RendererAPI_win32_opengl::SetSwapInterval(int interval)
{
    if (WGLExtensionSupported("WGL_EXT_swap_control"))
    {
        // Extension is supported, init pointers.
        PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");

        // this is another function from WGL_EXT_swap_control extension
        PFNWGLGETSWAPINTERVALEXTPROC wglGetSwapIntervalEXT = (PFNWGLGETSWAPINTERVALEXTPROC)wglGetProcAddress("wglGetSwapIntervalEXT");

        wglSwapIntervalEXT(interval);
        return true;
    }

    return false;
}

// Got from https://stackoverflow.com/questions/589064/how-to-enable-vertical-sync-in-opengl/589232
bool RendererAPI_win32_opengl::WGLExtensionSupported(const char* extension_name)
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





















//#include <windows.h>
//#include <GL/gl.h>
//HWND hWindow;
//HDC hDC;
//HGLRC hRC;
//struct Vertex
//{	
//float x, y, z;	
//float r, g, b;
//};
//LRESULT WINAPI WndProc( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam )
//{	
//    switch( msg )	{	
//        case WM_CLOSE:		
//            PostQuitMessage( 0 );		
//            return 0;	
//        case WM_PAINT:		
//            ValidateRect( hWnd, NULL );		
//            return 0;	
//    }	
//    return DefWindowProc( hWnd, msg, wParam, lParam );
//}
//
//int main()
//{	
//    HINSTANCE hInstance = GetModuleHandle( NULL );	
//    //create a window	
//    WNDCLASSEX wc = 
//    { 
//        sizeof(WNDCLASSEX), 
//        CS_CLASSDC, 
//        WndProc, 0, 0, hInstance, NULL, NULL, NULL, NULL, "MiniOGL", NULL 
//    };	
//    RegisterClassEx( &wc );	
//    hWindow = CreateWindowEx( WS_EX_APPWINDOW | WS_EX_WINDOWEDGE, "MiniOGL", "MiniOGL", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 		300, 300, NULL, NULL, hInstance, NULL );	
//    ShowWindow( hWindow, SW_SHOW );	
//    UpdateWindow( hWindow );	
//
//    //Set up pixel format and context	
//    hDC = GetDC( hWindow );	
//    PIXELFORMATDESCRIPTOR pfd = {0};	
//    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);	
//    pfd.nVersion = 1;	
//    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;	
//    pfd.iPixelType = PFD_TYPE_RGBA;	
//    pfd.cColorBits = 32;	
//    pfd.cDepthBits = 16;	
//    pfd.cStencilBits = 0;	
//    int pixelFormat = ChoosePixelFormat( hDC, &pfd );	
//    SetPixelFormat( hDC, pixelFormat, &pfd );	
//    hRC = wglCreateContext( hDC );	
//    wglMakeCurrent( hDC, hRC );	
//
//    //now create our triangle	
//    Vertex Triangle[3] = {{ 0.0f, 0.9f, 0.5f, 1.0f, 0.0f, 0.0f },{ -0.9f, -0.9f, 0.5f, 0.0f, 1.0f, 0.0f },{ 0.9f, -0.9f, 0.5f, 0.0f, 0.0f, 1.0f }};	
//    
//    MSG msg;	
//    bool RunApp = true;	
//    while( RunApp )	{		
//        if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )		
//        {			
//            if( msg.message == WM_QUIT )				
//                RunApp = false;			
//            TranslateMessage( &msg );			
//            DispatchMessage( &msg );		
//        }		
//        //render stuff		
//        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );		
//        glBegin( GL_TRIANGLES );		
//        for( int v = 0; v < 3; ++v )		
//        {			
//            glColor3f( Triangle[v].r, Triangle[v].g, Triangle[v].b );			
//            glVertex3f( Triangle[v].x, Triangle[v].y, Triangle[v].z );		
//        }		
//        glEnd();		
//        SwapBuffers( hDC );	
//    }	
//    wglMakeCurrent( 0, 0 );	
//    wglDeleteContext( hRC );	
//    ReleaseDC( hWindow, hDC );	
//    DestroyWindow( hWindow );	
//    return 0;
//}
//
//
//
//
//#include <windows.h>
//#include <d3d9.h>
//HWND hWindow;
//IDirect3D9* D3D;
//IDirect3DDevice9* Device;
//struct Vertex
//{	
//    float x, y, z, rhw;	
//    D3DCOLOR Color;
//};
//
//#define FVF_VERTEX (D3DFVF_XYZRHW | D3DFVF_DIFFUSE)
//LRESULT WINAPI WndProc( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam )
//{	
//    switch( msg )	
//    {	
//        case WM_CLOSE:		
//            PostQuitMessage( 0 );		
//            return 0;	
//        case WM_PAINT:		
//            ValidateRect( hWnd, NULL );		
//            return 0;	
//    }	
//    return DefWindowProc( hWnd, msg, wParam, lParam );
//}
//int main()
//{	
//    HINSTANCE hInstance = GetModuleHandle( NULL );	
//    //create a window	
//    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0, 0, hInstance, NULL, NULL, NULL, NULL, "MiniD3D", NULL };	
//    RegisterClassEx( &wc );	
//    hWindow = CreateWindowEx( WS_EX_APPWINDOW | WS_EX_WINDOWEDGE, "MiniD3D", "MiniD3D", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 300, 300, NULL, NULL, hInstance, NULL );	
//    ShowWindow( hWindow, SW_SHOW );	
//    UpdateWindow( hWindow );	
//
//    //Set up d3d	
//    D3D = Direct3DCreate9( D3D_SDK_VERSION );	
//    D3DPRESENT_PARAMETERS d3dpp = { 0 };	
//    d3dpp.Windowed = TRUE;	
//    d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;	
//    d3dpp.BackBufferFormat = D3DFMT_UNKNOWN;	
//    D3D->CreateDevice( D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hWindow, D3DCREATE_HARDWARE_VERTEXPROCESSING, &d3dpp, &Device );	
//    
//    //now create our triangle	
//    Vertex Triangle[3] = {{ 150.0f, 50.0f, 0.5f, 1.0f, D3DCOLOR_XRGB( 255, 0, 0 ) }, { 250.0f, 250.0f, 0.5f, 1.0f, D3DCOLOR_XRGB( 0, 255, 0 ) }, { 50.0f, 250.0f, 0.5f, 1.0f, D3DCOLOR_XRGB( 0, 0, 255 ) }	};	
//    MSG msg;	
//    bool RunApp = true;	
//    while( RunApp )	
//    {		
//        if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )		
//        {			
//            if( msg.message == WM_QUIT )				
//                RunApp = false;			
//            TranslateMessage( &msg );			
//            DispatchMessage( &msg );
//        }		
//        //render stuff		
//        Device->Clear( 0, 0, D3DCLEAR_TARGET, D3DCOLOR_XRGB( 0, 0, 0 ), 1.0f, 0 );		
//        Device->BeginScene();		
//        Device->SetFVF( FVF_VERTEX );		
//        Device->DrawPrimitiveUP( D3DPT_TRIANGLELIST, 1, Triangle, sizeof(Vertex) );		
//        Device->EndScene();		
//        Device->Present( 0, 0, 0, 0 );	
//    }	
//    Device->Release();	
//    D3D->Release();	
//    DestroyWindow( hWindow );	
//    return 0;
//}