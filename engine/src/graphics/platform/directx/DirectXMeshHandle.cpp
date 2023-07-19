#include "../../../../include/graphics/platform/directx/DirectXMeshHandle.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"

#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

#include <assert.h>

using namespace PhysicsEngine;

DirectXMeshHandle::DirectXMeshHandle()
{
}

DirectXMeshHandle::~DirectXMeshHandle()
{
}

void DirectXMeshHandle::addVertexBuffer(VertexBuffer *buffer, AttribType type, bool instanceBuffer)
{
    assert(buffer != nullptr);

    mBuffers.push_back(buffer);
}

void DirectXMeshHandle::addIndexBuffer(IndexBuffer *buffer)
{
    assert(buffer != nullptr);

    mIndexBuffer = buffer;
}

void DirectXMeshHandle::bind()
{
}

void DirectXMeshHandle::unbind()
{
}

void DirectXMeshHandle::drawLines(size_t vertexOffset, size_t vertexCount)
{
}

void DirectXMeshHandle::draw(size_t vertexOffset, size_t vertexCount)
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    for (size_t i = 0; i < mBuffers.size(); i++)
    {
        VertexBuffer *buffer = mBuffers[i];
        buffer->bind();

        context->Draw((unsigned int)vertexCount, (unsigned int)vertexOffset);

        buffer->unbind();
    }
}

void DirectXMeshHandle::drawIndexed(size_t indexOffset, size_t indexCount)
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    for (size_t i = 0; i < mBuffers.size(); i++)
    {
        VertexBuffer *buffer = mBuffers[i];
        buffer->bind();

        context->DrawIndexed((unsigned int)indexCount, (unsigned int)indexOffset, 0);

        buffer->unbind();
    }
}

void DirectXMeshHandle::drawInstanced(size_t vertexOffset, size_t vertexCount, size_t instanceCount)
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    for (size_t i = 0; i < mBuffers.size(); i++)
    {
        VertexBuffer *buffer = mBuffers[i];
        buffer->bind();

        context->DrawInstanced((unsigned int)vertexCount, (unsigned int)instanceCount, (unsigned int)vertexOffset, 0);

        buffer->unbind();
    }
}

void DirectXMeshHandle::drawIndexedInstanced(size_t indexOffset, size_t indexCount, size_t instanceCount)
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    for (size_t i = 0; i < mBuffers.size(); i++)
    {
        VertexBuffer *buffer = mBuffers[i];
        buffer->bind();

        context->DrawIndexedInstanced((unsigned int)indexCount, (unsigned int)instanceCount, (unsigned int)indexOffset,
                                      0, 0);

        buffer->unbind();
    }
}