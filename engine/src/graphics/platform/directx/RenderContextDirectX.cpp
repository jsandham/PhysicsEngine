#include "../../../../include/graphics/platform/directx/RenderContextDirectX.h"

#pragma comment(lib, "d3d11.lib")

#include <stdio.h>

using namespace PhysicsEngine;

RenderContextDirectX::RenderContextDirectX(void* window)
{
    DXGI_SWAP_CHAIN_DESC sd = { 0 };
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 0;
    sd.BufferDesc.RefreshRate.Denominator = 0;
    sd.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
    sd.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 1;
    sd.OutputWindow = static_cast<HWND>(window);
    sd.Windowed = true;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    sd.Flags = 0;

    D3D11CreateDeviceAndSwapChain(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &sd,
        &mSwapChain,
        &mDevice,
        nullptr,
        &mDeviceContext
    );

    ID3D11Resource* backbuffer = nullptr;
    mSwapChain->GetBuffer(0, __uuidof(ID3D11Resource), reinterpret_cast<void**>(&backbuffer));
    mDevice->CreateRenderTargetView(backbuffer, nullptr, &mTarget);
    backbuffer->Release();
}

RenderContextDirectX::~RenderContextDirectX()
{
    if (mDevice != nullptr)
    {
        mDevice->Release();
    }
    if (mDeviceContext != nullptr)
    {
        mDeviceContext->Release();
    }
    if (mSwapChain != nullptr)
    {
        mSwapChain->Release();
    }
}


void RenderContextDirectX::present()
{
    float color[] = {1, 0, 0, 1};
    mDeviceContext->ClearRenderTargetView(mTarget, color);

    mSwapChain->Present(1u, 0u);
}

void RenderContextDirectX::turnVsyncOn()
{

}

void RenderContextDirectX::turnVsyncOff()
{

}


//#include "../../../include/core/platform/RenderContext_win32_directx.h"
//
//#pragma comment(lib, "d3d11.lib")
//
//#include <stdio.h>
//
//using namespace PhysicsEngine;
//
//RenderContext_win32_directx::RenderContext_win32_directx()
//{
//}
//
//RenderContext_win32_directx::~RenderContext_win32_directx()
//{
//}
//
//void RenderContext_win32_directx::init(void* window)
//{
//    DXGI_SWAP_CHAIN_DESC sd = { 0 };
//    sd.BufferDesc.Width = 0;
//    sd.BufferDesc.Height = 0;
//    sd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
//    sd.BufferDesc.RefreshRate.Numerator = 0;
//    sd.BufferDesc.RefreshRate.Denominator = 0;
//    sd.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
//    sd.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
//    sd.SampleDesc.Count = 1;
//    sd.SampleDesc.Quality = 0;
//    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
//    sd.BufferCount = 1;
//    sd.OutputWindow = static_cast<HWND>(window);
//    sd.Windowed = true;
//    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
//    sd.Flags = 0;
//
//    D3D11CreateDeviceAndSwapChain(
//        nullptr,
//        D3D_DRIVER_TYPE_HARDWARE,
//        nullptr,
//        0,
//        nullptr,
//        0,
//        D3D11_SDK_VERSION,
//        &sd,
//        &mSwapChain,
//        &mDevice,
//        nullptr,
//        &mDeviceContext
//    );
//
//    ID3D11Resource* backbuffer = nullptr;
//    mSwapChain->GetBuffer(0, __uuidof(ID3D11Resource), reinterpret_cast<void**>(&backbuffer));
//    mDevice->CreateRenderTargetView(backbuffer, nullptr, &mTarget);
//    backbuffer->Release();
//}
//
//void RenderContext_win32_directx::update()
//{
//    float color[] = { 1, 0, 0, 1 };
//    mDeviceContext->ClearRenderTargetView(mTarget, color);
//
//    mSwapChain->Present(1u, 0u);
//}
//
//void RenderContext_win32_directx::cleanup()
//{
//    if (mDevice != nullptr)
//    {
//        mDevice->Release();
//    }
//    if (mDeviceContext != nullptr)
//    {
//        mDeviceContext->Release();
//    }
//    if (mSwapChain != nullptr)
//    {
//        mSwapChain->Release();
//    }
//}
//
//void RenderContext_win32_directx::turnVsyncOn()
//{
//
//}
//
//void RenderContext_win32_directx::turnVsyncOff()
//{
//
//}
