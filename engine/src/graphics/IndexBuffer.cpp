#include "../../include/graphics/IndexBuffer.h"
#include "../../include/graphics/RenderContext.h"

#include "../../include/graphics/platform/directx/DirectXIndexBuffer.h"
#include "../../include/graphics/platform/opengl/OpenGLIndexBuffer.h"

using namespace PhysicsEngine;

IndexBuffer::IndexBuffer() : mSize(0)
{
}

IndexBuffer::~IndexBuffer()
{
}

size_t IndexBuffer::getSize() const
{
    return mSize;
}

IndexBuffer *IndexBuffer::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLIndexBuffer();
    case RenderAPI::DirectX:
        return new DirectXIndexBuffer();
    }

    return nullptr;
}