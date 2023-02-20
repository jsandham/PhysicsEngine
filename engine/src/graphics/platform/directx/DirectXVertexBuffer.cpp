#include "../../../../include/graphics/platform/directx/DirectXVertexBuffer.h"

#include <assert.h>

using namespace PhysicsEngine;

DirectXVertexBuffer::DirectXVertexBuffer()
{
}

DirectXVertexBuffer::~DirectXVertexBuffer()
{
}

void DirectXVertexBuffer::resize(size_t size)
{
 
}

void DirectXVertexBuffer::setData(const void* data, size_t offset, size_t size)
{
	assert(size <= mSize);

}

void DirectXVertexBuffer::bind()
{
    
}

void DirectXVertexBuffer::unbind()
{
   
}

void* DirectXVertexBuffer::getBuffer()
{
	return nullptr;// static_cast<void*>(&mBuffer);
}