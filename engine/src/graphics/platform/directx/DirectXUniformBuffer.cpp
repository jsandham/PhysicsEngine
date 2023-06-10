#include "../../../../include/graphics/platform/directx/DirectXUniformBuffer.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"

#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

#include <assert.h>

using namespace PhysicsEngine;

DirectXUniformBuffer::DirectXUniformBuffer(size_t size, unsigned int bindingPoint) : UniformBuffer()
{
    mSize = size;
    mBindingPoint = bindingPoint;

    ZeroMemory(&mBufferDesc, sizeof(D3D11_BUFFER_DESC));
    mBufferDesc.Usage = D3D11_USAGE_DYNAMIC;             // write access access by CPU and GPU
    mBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;  // use as a constant buffer
    mBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE; // allow CPU to write in buffer

    ID3D11Device *device = DirectXRenderContext::get()->getD3DDevice();

    assert(device != nullptr);

    CHECK_ERROR(device->CreateBuffer(&mBufferDesc, NULL, &mBuffer));
}

DirectXUniformBuffer::~DirectXUniformBuffer()
{
    mBuffer->Release();
}

size_t DirectXUniformBuffer::getSize() const
{
    return mSize;
}

unsigned int DirectXUniformBuffer::getBindingPoint() const
{
    return mBindingPoint;
}

void DirectXUniformBuffer::bind(PipelineStage stage)
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);
    
    switch (stage)
    {
    case PipelineStage::VS:
        context->VSSetConstantBuffers(0, 1, &mBuffer);
        break;
    case PipelineStage::PS:
        context->PSSetConstantBuffers(0, 1, &mBuffer);
        break;
    }
}

void DirectXUniformBuffer::unbind()
{
}

void DirectXUniformBuffer::setData(const void *data, size_t offset, size_t size)
{
    assert(data != NULL);
    assert(offset + size <= mSize);

    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    CHECK_ERROR(context->Map(mBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mMappedSubresource));
    memcpy(mMappedSubresource.pData, static_cast<const char*>(data) + offset, size);
    context->Unmap(mBuffer, 0);
}