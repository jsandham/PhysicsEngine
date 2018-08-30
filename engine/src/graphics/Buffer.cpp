#include "../../include/graphics/Buffer.h"

using namespace PhysicsEngine;

Buffer::Buffer()
{
	
}

Buffer::~Buffer()
{
	
}

void Buffer::generate(GLenum target, GLenum usage)
{
	this->target = target;
	this->usage = usage;

	glGenBuffers(1, &handle);
}

void Buffer::destroy()
{
	glDeleteBuffers(1, &handle);
}

void Buffer::bind()
{
	glBindBuffer(target, handle);
}

void Buffer::unbind()
{
	glBindBuffer(target, 0);
}

void Buffer::setData(const void* data, std::size_t size)
{
	glBufferData(target, size, data, usage);
}

void Buffer::setSubData(const void* data, unsigned int offset, std::size_t size)
{
	glBufferSubData(target, offset, size, data);
}

void Buffer::setRange(unsigned int index, unsigned int offset, std::size_t size)
{
	glBindBufferRange(target, index, handle, offset, size);
}