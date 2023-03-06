#include "../../../../include/graphics/platform/directx/DirectXVertexBuffer.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"

#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

#include <assert.h>

using namespace PhysicsEngine;

DirectXVertexBuffer::DirectXVertexBuffer()
{
    ZeroMemory(&mBufferDesc, sizeof(D3D11_BUFFER_DESC));
    mBufferDesc.Usage = D3D11_USAGE_DYNAMIC;               // write access access by CPU and GPU
    mBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;      // use as a vertex buffer
    mBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;   // allow CPU to write in buffer
}

DirectXVertexBuffer::~DirectXVertexBuffer()
{
}

void DirectXVertexBuffer::resize(size_t size)
{
    mSize = size;
    mBufferDesc.ByteWidth = (unsigned int)size;

    ID3D11Device *device = DirectXRenderContext::get()->getD3DDevice();

    assert(device != nullptr);

    //CHECK_ERROR(device->CreateBuffer(&mBufferDesc, NULL, &mBuffer));
}

void DirectXVertexBuffer::setData(const void* data, size_t offset, size_t size)
{
	assert(size <= mSize);

    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    //CHECK_ERROR(context->Map(mBuffer, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &mMappedSubresource));
    //memcpy(mMappedSubresource.pData, data, size);
    //context->Unmap(mBuffer, NULL);

}

void DirectXVertexBuffer::bind()
{
    //unsigned int offset = 0;
    //unsigned int stride = m_Layout.GetStride();

    //ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    //assert(context != nullptr);

    //context->IASetInputLayout(m_InputLayout);
    //context->IASetVertexBuffers(0, 1, &m_BufferHandle, &stride, &offset);
}

void DirectXVertexBuffer::unbind()
{
   
}

void* DirectXVertexBuffer::getBuffer()
{
    return nullptr;//static_cast<void *>(&mBuffer);
}