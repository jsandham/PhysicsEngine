#include "../../../../include/graphics/platform/directx/DirectXVertexBuffer.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"

#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

#include <assert.h>

using namespace PhysicsEngine;

DirectXVertexBuffer::DirectXVertexBuffer()
{
    ZeroMemory(&mBufferDesc, sizeof(D3D11_BUFFER_DESC));
    mBufferDesc.Usage = D3D11_USAGE_DYNAMIC;             // write access access by CPU and GPU
    mBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;    // use as a vertex buffer
    mBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE; // allow CPU to write in buffer

    mBuffer = NULL;
}

DirectXVertexBuffer::~DirectXVertexBuffer()
{
    mBuffer->Release();
}

void DirectXVertexBuffer::resize(size_t size)
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

void DirectXVertexBuffer::setData(const void *data, size_t offset, size_t size)
{
    assert(mBuffer != NULL);
    assert(data != NULL);
    assert(offset + size <= mSize);

    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    CHECK_ERROR(context->Map(mBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mMappedSubresource));
    memcpy(mMappedSubresource.pData, static_cast<const char *>(data) + offset, size);
    context->Unmap(mBuffer, 0);
}

void DirectXVertexBuffer::bind()
{
    // unsigned int offset = 0;
    // unsigned int stride = m_Layout.GetStride();

    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    // context->IASetInputLayout(m_InputLayout);
    // context->IASetVertexBuffers(0, 1, &m_BufferHandle, &stride, &offset);
}

void DirectXVertexBuffer::unbind()
{
    // ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    // assert(context != nullptr);

    // context->IASetInputLayout(NULL);
    // context->IASetVertexBuffers(0, 1, NULL, NULL, NULL);
}

void *DirectXVertexBuffer::getBuffer()
{
    return nullptr; // static_cast<void *>(&mBuffer);
}