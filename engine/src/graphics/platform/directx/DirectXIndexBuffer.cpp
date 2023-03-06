#include "../../../../include/graphics/platform/directx/DirectXIndexBuffer.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"

#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

#include <assert.h>

using namespace PhysicsEngine;

DirectXIndexBuffer::DirectXIndexBuffer()
{
    ZeroMemory(&mBufferDesc, sizeof(D3D11_BUFFER_DESC));
    mBufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
    mBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    mBufferDesc.CPUAccessFlags = 0;
    mBufferDesc.MiscFlags = 0;
}

DirectXIndexBuffer::~DirectXIndexBuffer()
{
}

void DirectXIndexBuffer::resize(size_t size)
{
    mSize = size;
    mBufferDesc.ByteWidth = (unsigned int)size;

    ID3D11Device *device = DirectXRenderContext::get()->getD3DDevice();
    assert(device != nullptr);

    //CHECK_ERROR(device->CreateBuffer(&mBufferDesc, NULL, &mBuffer));
}

void DirectXIndexBuffer::setData(void* data, size_t offset, size_t size)
{
	assert(size <= mSize);

    ID3D11Device *device = DirectXRenderContext::get()->getD3DDevice();
    assert(device != nullptr);

    //D3D11_SUBRESOURCE_DATA ibInitData;
    //ibInitData.pSysMem = data;
    //CHECK_ERROR(device->CreateBuffer(&mBufferDesc, &ibInitData, &mBuffer));
}

void DirectXIndexBuffer::bind()
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    //context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    //context->IASetIndexBuffer(mBuffer, DXGI_FORMAT_R32_UINT, 0);
}

void DirectXIndexBuffer::unbind()
{

}

void* DirectXIndexBuffer::getBuffer()
{
    return nullptr;//static_cast<void *>(&mBuffer);
}