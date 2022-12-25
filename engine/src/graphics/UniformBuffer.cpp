#include "../../include/graphics/UniformBuffer.h"
#include "../../include/graphics/RenderContext.h"

#include "../../include/graphics/platform/opengl/OpenGLUniformBuffer.h"
#include "../../include/graphics/platform/directx/DirectXUniformBuffer.h"

using namespace PhysicsEngine;

UniformBuffer::UniformBuffer()
{
}

UniformBuffer::~UniformBuffer()
{
}

UniformBuffer *UniformBuffer::create(size_t size, unsigned int bindingPoint)
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLUniformBuffer(size, bindingPoint);
    case RenderAPI::DirectX:
        return new DirectXUniformBuffer(size, bindingPoint);
    }

    return nullptr;
}