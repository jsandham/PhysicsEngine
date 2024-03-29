#include "../../../../include/graphics/platform/opengl/OpenGLUniformBuffer.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"

#include <GL/glew.h>
#include <assert.h>
#include <cstring>

using namespace PhysicsEngine;

OpenGLUniformBuffer::OpenGLUniformBuffer(size_t size, unsigned int bindingPoint)
{
    mSize = size;
    mBindingPoint = bindingPoint;

    assert(mSize <= 2048);

    memset(&mData, 0, 2048);

    CHECK_ERROR(glGenBuffers(1, &mBuffer));
    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, mBuffer));
    CHECK_ERROR(glBufferData(GL_UNIFORM_BUFFER, mSize, NULL, GL_DYNAMIC_DRAW));
    CHECK_ERROR(glBindBufferRange(GL_UNIFORM_BUFFER, mBindingPoint, mBuffer, 0, mSize));
    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, 0));
}

OpenGLUniformBuffer::~OpenGLUniformBuffer()
{
    CHECK_ERROR(glDeleteBuffers(1, &mBuffer));
}

size_t OpenGLUniformBuffer::getSize() const
{
    return mSize;
}

unsigned int OpenGLUniformBuffer::getBindingPoint() const
{
    return mBindingPoint;
}

void OpenGLUniformBuffer::bind(PipelineStage stage)
{
    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, mBuffer));
}

void OpenGLUniformBuffer::unbind(PipelineStage stage)
{
    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, 0));
}

void OpenGLUniformBuffer::setData(const void *data, size_t offset, size_t size)
{
    assert(data != NULL);
    assert(offset + size <= mSize);

    memcpy(mData + offset, data, size);
}

void OpenGLUniformBuffer::getData(void *data, size_t offset, size_t size)
{
    assert(data != NULL);
    assert(offset + size <= mSize);

    memcpy(data, mData + offset, size);
}

void OpenGLUniformBuffer::copyDataToDevice()
{
    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 0, mSize, mData));
}
