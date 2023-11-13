#include <algorithm>
#include <assert.h>
#include <iostream>
#include <random>

#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/directx/DirectXRenderer.h"

using namespace PhysicsEngine;

#define CHECK_ERROR_IMPL(ROUTINE, LINE, FILE)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        HRESULT hr = ROUTINE;                                                                                          \
        LPTSTR lpBuf = NULL;                                                                                           \
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,     \
                      NULL, hr, 0, (LPTSTR)&lpBuf, 0, NULL);                                                           \
    } while (0)

#define CHECK_ERROR(ROUTINE) CHECK_ERROR_IMPL(ROUTINE, std::to_string(__LINE__), std::string(__FILE__))

void DirectXRenderer::init_impl()
{
    mContext = DirectXRenderContext::get();

    DXGI_ADAPTER_DESC descr;
    mContext->getAdapter()->GetDesc(&descr);

    // Log::warn(("Vender: " + vender + "\n").c_str());
    Log::warn(("Dedicated video memory: " + std::to_string(descr.DedicatedVideoMemory) + "\n").c_str());
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