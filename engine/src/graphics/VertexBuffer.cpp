#include "../../include/graphics/VertexBuffer.h"
#include "../../include/graphics/RenderContext.h"

#include "../../include/graphics/platform/directx/DirectXVertexBuffer.h"
#include "../../include/graphics/platform/opengl/OpenGLVertexBuffer.h"

using namespace PhysicsEngine;

VertexBuffer::VertexBuffer() : mSize(0)
{
}

VertexBuffer::~VertexBuffer()
{
}

size_t VertexBuffer::getSize() const
{
    return mSize;
}

VertexBuffer *VertexBuffer::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLVertexBuffer();
    case RenderAPI::DirectX:
        return new DirectXVertexBuffer();
    }

    return nullptr;
}