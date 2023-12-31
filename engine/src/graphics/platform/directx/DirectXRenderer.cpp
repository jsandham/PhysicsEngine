#include <algorithm>
#include <assert.h>
#include <iostream>
#include <random>

#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"
#include "../../../../include/graphics/platform/directx/DirectXRenderer.h"

using namespace PhysicsEngine;

void DirectXRenderer::init_impl()
{
    DXGI_ADAPTER_DESC descr;
    mContext->getAdapter()->GetDesc(&descr);

    // Log::warn(("Vender: " + vender + "\n").c_str());
    Log::warn(("Dedicated video memory: " + std::to_string(descr.DedicatedVideoMemory) + "\n").c_str());
}

DirectXRenderer::DirectXRenderer()
{
    mContext = DirectXRenderContext::get();

    ZeroMemory(&mRasterizerDescr, sizeof(D3D11_RASTERIZER_DESC));
    mRasterizerDescr.FillMode = D3D11_FILL_SOLID;
    mRasterizerDescr.CullMode = D3D11_CULL_FRONT;
    mRasterizerDescr.DepthClipEnable = true;
    mRasterizerDescr.MultisampleEnable = true;

    CHECK_ERROR(mContext->getD3DDevice()->CreateRasterizerState(&mRasterizerDescr, &mRasterizerState));

    mContext->getD3DDeviceContext()->RSSetState(mRasterizerState);

    ZeroMemory(&mBlendDescr, sizeof(D3D11_BLEND_DESC));

    D3D11_RENDER_TARGET_BLEND_DESC rtBlend;
    rtBlend.BlendEnable = TRUE;
    rtBlend.SrcBlend = D3D11_BLEND_ONE;
    rtBlend.DestBlend = D3D11_BLEND_ZERO;
    rtBlend.BlendOp = D3D11_BLEND_OP_ADD;
    rtBlend.SrcBlendAlpha = D3D11_BLEND_ONE;
    rtBlend.DestBlendAlpha = D3D11_BLEND_ZERO;
    rtBlend.BlendOpAlpha = D3D11_BLEND_OP_ADD;
    rtBlend.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

    for (auto &i : mBlendDescr.RenderTarget)
    {
        i = rtBlend;
    }

    CHECK_ERROR(mContext->getD3DDevice()->CreateBlendState(&mBlendDescr, &mBlendState));

    mContext->getD3DDeviceContext()->OMSetBlendState(mBlendState, NULL, 0xffffffff);
}

DirectXRenderer::~DirectXRenderer()
{
    mRasterizerState->Release();
    mBlendState->Release();
}

void DirectXRenderer::present_impl()
{
    mContext->present();
}

void DirectXRenderer::turnVsyncOn_impl()
{
    mContext->turnVsyncOn();
}

void DirectXRenderer::turnVsyncOff_impl()
{
    mContext->turnVsyncOff();
}

void DirectXRenderer::bindBackBuffer_impl()
{
    mContext->bindBackBuffer();
}

void DirectXRenderer::unbindBackBuffer_impl()
{
    mContext->unBindBackBuffer();
}

void DirectXRenderer::clearBackBufferColor_impl(const Color &color)
{
    this->clearBackBufferColor_impl(color.mR, color.mG, color.mB, color.mA);
}

void DirectXRenderer::clearBackBufferColor_impl(float r, float g, float b, float a)
{
    mContext->clearBackBufferColor(r, g, b, a);
}

void DirectXRenderer::setViewport_impl(int x, int y, int width, int height)
{
    D3D11_VIEWPORT viewport = {0};

    viewport.TopLeftX = static_cast<float>(x);
    viewport.TopLeftY = static_cast<float>(y);
    viewport.Width = static_cast<float>(width);
    viewport.Height = static_cast<float>(height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;

    mContext->getD3DDeviceContext()->RSSetViewports(1, &viewport);
}

void DirectXRenderer::setScissor_impl(int x, int y, int width, int height)
{
    D3D11_RECT rect = {0};

    rect.left = x;
    rect.top = y;
    rect.right = x + width;
    rect.bottom = y + height;

    mContext->getD3DDeviceContext()->RSSetScissorRects(1, &rect);
}

void DirectXRenderer::turnOn_impl(Capability capability)
{
}
void DirectXRenderer::turnOff_impl(Capability capability)
{
}
void DirectXRenderer::setBlending_impl(BlendingFactor source, BlendingFactor dest)
{
}
void DirectXRenderer::draw_impl(MeshHandle *meshHandle, size_t vertexOffset, size_t vertexCount, TimingQuery &query)
{
    meshHandle->draw(vertexOffset, vertexCount);
};
void DirectXRenderer::drawIndexed_impl(MeshHandle *meshHandle, size_t indexOffset, size_t indexCount, TimingQuery &query)
{
    meshHandle->drawIndexed(indexOffset, indexCount);
};

void DirectXRenderer::drawInstanced_impl(MeshHandle *meshHandle, size_t vertexOffset, size_t vertexCount, size_t instanceCount, TimingQuery &query)
{
    meshHandle->drawInstanced(vertexOffset, vertexCount, instanceCount);
};

void DirectXRenderer::drawIndexedInstanced_impl(MeshHandle *meshHandle, size_t indexOffset, size_t indexCount, size_t instanceCount, TimingQuery &query)
{
    meshHandle->drawIndexedInstanced(indexOffset, indexCount, instanceCount);
};

void DirectXRenderer::beginQuery_impl(unsigned int queryId)
{
}
void DirectXRenderer::endQuery_impl(unsigned int queryId, unsigned long long *elapsedTime)
{
}