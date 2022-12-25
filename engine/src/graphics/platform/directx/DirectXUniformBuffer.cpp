#include "../../../../include/graphics/platform/directx/DirectXUniformBuffer.h"

#include<assert.h>

using namespace PhysicsEngine;

DirectXUniformBuffer::DirectXUniformBuffer(size_t size, unsigned int bindingPoint) : UniformBuffer()
{
    mSize = size;
    mBindingPoint = bindingPoint;
}

DirectXUniformBuffer::~DirectXUniformBuffer()
{
}

size_t DirectXUniformBuffer::getSize() const
{
    return mSize;
}

unsigned int DirectXUniformBuffer::getBindingPoint() const
{
    return mBindingPoint;
}

void DirectXUniformBuffer::bind()
{
}

void DirectXUniformBuffer::unbind()
{
}

void DirectXUniformBuffer::setData(void *data, size_t offset, size_t size)
{
    assert(offset + size <= mSize);
}