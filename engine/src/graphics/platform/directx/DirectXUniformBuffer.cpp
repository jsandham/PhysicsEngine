#include "../../../../include/graphics/platform/directx/DirectXUniformBuffer.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"

#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

#include <assert.h>
#include <vector>

using namespace PhysicsEngine;

DirectXUniformBuffer::DirectXUniformBuffer(size_t size, unsigned int bindingPoint) : UniformBuffer()
{
    // Constant buffer size must be multiple of 16 bytes
    mSize = 16 * ((size - 1) / 16 + 1);
    mBindingPoint = bindingPoint;

    assert(mSize <= 2048);

    ZeroMemory(&mBufferDesc, sizeof(D3D11_BUFFER_DESC));
    mBufferDesc.Usage = D3D11_USAGE_DYNAMIC;             // write access access by CPU and GPU
    mBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;  // use as a constant buffer
    mBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE; // allow CPU to write in buffer
    mBufferDesc.ByteWidth = static_cast<unsigned int>(mSize);

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
        context->VSSetConstantBuffers(mBindingPoint, 1, &mBuffer);
        break;
    case PipelineStage::PS:
        context->PSSetConstantBuffers(mBindingPoint, 1, &mBuffer);
        break;
    }
}
void DirectXUniformBuffer::unbind(PipelineStage stage)
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    ID3D11Buffer* mNullBuffer = nullptr;

    switch (stage)
    {
    case PipelineStage::VS:
        context->VSSetConstantBuffers(mBindingPoint, 1, &mNullBuffer);
        break;
    case PipelineStage::PS:
        context->PSSetConstantBuffers(mBindingPoint, 1, &mNullBuffer);
        break;
    }
}

void DirectXUniformBuffer::setData(const void *data, size_t offset, size_t size)
{
    assert(data != NULL);
    assert(offset + size <= mSize);

    memcpy(mData + offset, data, size);
}

void DirectXUniformBuffer::getData(void *data, size_t offset, size_t size)
{
    assert(data != NULL);
    assert(offset + size <= mSize);

    memcpy(data, mData + offset, size);
}

void DirectXUniformBuffer::copyDataToDevice()
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    CHECK_ERROR(context->Map(mBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mMappedSubresource));
    memcpy(mMappedSubresource.pData, static_cast<const char *>(mData), mSize);
    context->Unmap(mBuffer, 0);
}