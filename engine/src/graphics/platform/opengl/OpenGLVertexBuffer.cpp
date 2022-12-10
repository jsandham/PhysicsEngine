#include "../../../../include/graphics/platform/opengl/OpenGLVertexBuffer.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"

#include <GL/glew.h>

using namespace PhysicsEngine;

OpenGLVertexBuffer::OpenGLVertexBuffer() : mSize(0)
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

	CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, mBuffer));
    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW));
}

//void OpenGLVertexBuffer::setData(const void* data, size_t size)
//{
//	assert(size <= mSize);
//
//	CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, mBuffer));
//	CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW));
//}

void OpenGLVertexBuffer::bind()
{
	CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, mBuffer));
}

void OpenGLVertexBuffer::unbind()
{
	CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void* OpenGLVertexBuffer::get()
{
	return static_cast<void*>(&mBuffer);
}