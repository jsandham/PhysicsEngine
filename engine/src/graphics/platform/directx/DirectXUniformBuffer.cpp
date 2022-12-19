#include "../../../../include/graphics/platform/directx/DirectXUniformBuffer.h"

using namespace PhysicsEngine;

DirectXUniformBuffer::DirectXUniformBuffer(size_t size) : UniformBuffer(size)
{
}

DirectXUniformBuffer::~DirectXUniformBuffer()
{
}

void DirectXUniformBuffer::setData(void *data, size_t size, size_t offset, size_t bindingPoint)
{
   
}