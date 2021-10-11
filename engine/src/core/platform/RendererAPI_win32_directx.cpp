#include "../../../include/core/platform/RendererAPI_win32_directx.h"

#include <stdio.h>

using namespace PhysicsEngine;

RendererAPI_win32_directx::RendererAPI_win32_directx()
{
}

RendererAPI_win32_directx::~RendererAPI_win32_directx()
{
}

void RendererAPI_win32_directx::init(void* window)
{
    // Set up d3d	
    D3D = Direct3DCreate9( D3D_SDK_VERSION );	
    D3DPRESENT_PARAMETERS d3dpp = { 0 };	
    d3dpp.Windowed = TRUE;	
    d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;	
    d3dpp.BackBufferFormat = D3DFMT_UNKNOWN;	
    D3D->CreateDevice( D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, static_cast<HWND>(window), D3DCREATE_HARDWARE_VERTEXPROCESSING, &d3dpp, &Device );
}

void RendererAPI_win32_directx::update()
{
    Device->Present(0, 0, 0, 0);
}

void RendererAPI_win32_directx::cleanup()
{
    Device->Release();
    D3D->Release();	
}
