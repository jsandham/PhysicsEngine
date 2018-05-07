#include <cstddef>
#include "UniformBufferObject.h"

using namespace PhysicsEngine;

UniformBufferObject::UniformBufferObject()
{

}

UniformBufferObject::~UniformBufferObject()
{

}

void UniformBufferObject::generate()
{
	glGenBuffers(1, &handle);
}

void UniformBufferObject::destroy()
{
	glDeleteBuffers(1, &handle);
}

void UniformBufferObject::bind()
{
	glBindBuffer(GL_UNIFORM_BUFFER, handle);
}

void UniformBufferObject::unbind()
{
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void UniformBufferObject::setData(const void* data, std::size_t size)
{
	glBufferData(GL_UNIFORM_BUFFER, size, data, GL_STATIC_DRAW);
}

void UniformBufferObject::setSubData(const void* data, unsigned int offset, std::size_t size)
{
	glBufferSubData(GL_UNIFORM_BUFFER, offset, size, data);
}

void UniformBufferObject::setRange(unsigned int offset, std::size_t size)
{
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, handle, offset, size);
}