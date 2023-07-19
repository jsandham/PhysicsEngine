#include "../../../../include/graphics/platform/opengl/OpenGLIndexBuffer.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"

#include <GL/glew.h>
#include <assert.h>

using namespace PhysicsEngine;

OpenGLIndexBuffer::OpenGLIndexBuffer()
{
    CHECK_ERROR(glGenBuffers(1, &mBuffer));
}

OpenGLIndexBuffer::~OpenGLIndexBuffer()
{
    CHECK_ERROR(glDeleteBuffers(1, &mBuffer));
}

void OpenGLIndexBuffer::resize(size_t size)
{
    mSize = size;
    CHECK_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW));
}

void OpenGLIndexBuffer::setData(void *data, size_t offset, size_t size)
{
    assert(offset + size <= mSize);

    CHECK_ERROR(glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, offset, size, data));
}

void OpenGLIndexBuffer::bind()
{
    CHECK_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mBuffer));
}

void OpenGLIndexBuffer::unbind()
{
    CHECK_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
}

void *OpenGLIndexBuffer::getBuffer()
{
    return static_cast<void *>(&mBuffer);
}