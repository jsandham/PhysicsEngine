#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

#pragma comment(lib, "d3d11.lib")

#include <stdio.h>

using namespace PhysicsEngine;

DirectXRenderContext::DirectXRenderContext(void* window)
{
    mSwapInterval = 1;

    DXGI_SWAP_CHAIN_DESC sd = { 0 };
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;// DXGI_FORMAT_B8G8R8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;// 0;
    sd.BufferDesc.RefreshRate.Denominator = 1;// 0;
    sd.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
    sd.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 2; //1;
    sd.OutputWindow = static_cast<HWND>(window);
    sd.Windowed = true;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    sd.Flags = 0;

    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &sd,
        &mSwapChain,
        &mD3DDevice,
        nullptr,
        &mD3DDeviceContext
    );

    if (S_OK == mD3DDevice->QueryInterface(__uuidof(IDXGIDevice), (void**)&mDevice))
    {
        mDevice->GetAdapter(&mAdapter);
    }

    LPTSTR lpBuf = NULL;
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, hr, 0, (LPTSTR)&lpBuf, 0, NULL);

    ID3D11Resource* backbuffer = nullptr;
    mSwapChain->GetBuffer(0, __uuidof(ID3D11Resource), reinterpret_cast<void**>(&backbuffer));
    mD3DDevice->CreateRenderTargetView(backbuffer, nullptr, &mD3DTarget);
    backbuffer->Release();

    mD3DDeviceContext->OMSetRenderTargets(1, &mD3DTarget, NULL);
}

DirectXRenderContext::~DirectXRenderContext()
{
    if (mD3DDevice != nullptr)
    {
        mD3DDevice->Release();
    }
    if (mD3DDeviceContext != nullptr)
    {
        mD3DDeviceContext->Release();
    }

    if (mDevice != nullptr)
    {
        mDevice->Release();
    }
    if (mAdapter != nullptr)
    {
        mAdapter->Release();
    }
    if (mSwapChain != nullptr)
    {
        mSwapChain->Release();
    }
}


void DirectXRenderContext::present()
{
    //float color[] = {1, 0, 0, 1};
    //mDeviceContext->ClearRenderTargetView(mTarget, color);

    HRESULT hr = mSwapChain->Present(mSwapInterval, 0);

    LPTSTR lpBuf = NULL;
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, hr, 0, (LPTSTR)&lpBuf, 0, NULL);
}

void DirectXRenderContext::turnVsyncOn()
{
    mSwapInterval = 1;
}

void DirectXRenderContext::turnVsyncOff()
{
    mSwapInterval = 0;
}