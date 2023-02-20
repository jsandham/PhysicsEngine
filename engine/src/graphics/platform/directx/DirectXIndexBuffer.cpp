#include "../../../../include/graphics/platform/directx/DirectXIndexBuffer.h"

#include <assert.h>

using namespace PhysicsEngine;

DirectXIndexBuffer::DirectXIndexBuffer()
{
}

DirectXIndexBuffer::~DirectXIndexBuffer()
{
}

void DirectXIndexBuffer::resize(size_t size)
{

}

void DirectXIndexBuffer::setData(void* data, size_t offset, size_t size)
{
	assert(size <= mSize);

}

void DirectXIndexBuffer::bind()
{

}

void DirectXIndexBuffer::unbind()
{

}

void* DirectXIndexBuffer::getBuffer()
{
	return nullptr;// static_cast<void*>(&mBuffer);
}