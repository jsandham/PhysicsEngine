#include "../../../../include/graphics/platform/opengl/OpenGLVertexBuffer.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"

#include <assert.h>
#include <GL/glew.h>

using namespace PhysicsEngine;

OpenGLVertexBuffer::OpenGLVertexBuffer()
{
	CHECK_ERROR(glGenBuffers(1, &mBuffer));
}

OpenGLVertexBuffer::~OpenGLVertexBuffer()
{
	CHECK_ERROR(glDeleteBuffers(1, &mBuffer));
}

void OpenGLVertexBuffer::resize(size_t size)
{
	mSize = size;
    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW));
}

void OpenGLVertexBuffer::setData(const void* data, size_t offset, size_t size)
{
    assert(data != NULL);
	assert(offset + size <= mSize);

	CHECK_ERROR(glBufferSubData(GL_ARRAY_BUFFER, offset, size, data));
}

void OpenGLVertexBuffer::bind()
{
	CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, mBuffer));
}

void OpenGLVertexBuffer::unbind()
{
	CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void* OpenGLVertexBuffer::getBuffer()
{
	return static_cast<void*>(&mBuffer);
}