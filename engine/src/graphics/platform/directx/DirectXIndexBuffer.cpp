#include "../../../../include/graphics/platform/directx/DirectXIndexBuffer.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"

#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

#include <assert.h>

using namespace PhysicsEngine;

DirectXIndexBuffer::DirectXIndexBuffer()
{
    ZeroMemory(&mBufferDesc, sizeof(D3D11_BUFFER_DESC));
    mBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
    mBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    mBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    mBufferDesc.MiscFlags = 0;

    mBuffer = NULL;
}

DirectXIndexBuffer::~DirectXIndexBuffer()
{
    mBuffer->Release();
}

void DirectXIndexBuffer::resize(size_t size)
{
    if (mBuffer != NULL)
    {
        mBuffer->Release();
    }

    mSize = size;
    mBufferDesc.ByteWidth = (unsigned int)size;

    ID3D11Device *device = DirectXRenderContext::get()->getD3DDevice();
    assert(device != nullptr);

    CHECK_ERROR(device->CreateBuffer(&mBufferDesc, NULL, &mBuffer));
}

void DirectXIndexBuffer::setData(const void *data, size_t offset, size_t size)
{
    assert(mBuffer != NULL);
    assert(data != NULL);
    assert(offset + size <= mSize);

    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    CHECK_ERROR(context->Map(mBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mMappedSubresource));
    memcpy(mMappedSubresource.pData, static_cast<const char*>(data) + offset, size);
    context->Unmap(mBuffer, 0);
}

void DirectXIndexBuffer::bind()
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    context->IASetIndexBuffer(mBuffer, DXGI_FORMAT_R32_UINT, 0);
}

void DirectXIndexBuffer::unbind()
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    context->IASetIndexBuffer(NULL, DXGI_FORMAT_R32_UINT, 0);
}

void *DirectXIndexBuffer::getBuffer()
{
    return nullptr; // static_cast<void *>(&mBuffer);
}