#include "../../../../include/graphics/platform/opengl/OpenGLUniformBuffer.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"

#include <GL/glew.h>

using namespace PhysicsEngine;

OpenGLUniformBuffer::OpenGLUniformBuffer(size_t size) : UniformBuffer(size)
{
    CHECK_ERROR(glGenBuffers(1, &mBuffer));
    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, mBuffer));
    CHECK_ERROR(glBufferData(GL_UNIFORM_BUFFER, size, NULL, GL_DYNAMIC_DRAW));
    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, 0));
}

OpenGLUniformBuffer::~OpenGLUniformBuffer()
{
    CHECK_ERROR(glDeleteBuffers(1, &mBuffer));
}

void OpenGLUniformBuffer::setData(void *data, size_t size, size_t offset, size_t bindingPoint)
{
    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, mBuffer));
    CHECK_ERROR(glBindBufferRange(GL_UNIFORM_BUFFER, bindingPoint, mBuffer, offset, size));
    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, 0));
}