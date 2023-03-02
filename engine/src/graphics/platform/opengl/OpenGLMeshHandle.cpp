#include "../../../../include/graphics/platform/opengl/OpenGLMeshHandle.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"

#include <GL/glew.h>
#include <glm/glm.hpp>

using namespace PhysicsEngine;

OpenGLMeshHandle::OpenGLMeshHandle()
{
    CHECK_ERROR(glGenVertexArrays(1, &mVao));
    mVertexAttribIndex = 0;
}

OpenGLMeshHandle::~OpenGLMeshHandle()
{
    CHECK_ERROR(glDeleteVertexArrays(1, &mVao));
}

void OpenGLMeshHandle::bind()
{
    CHECK_ERROR(glBindVertexArray(mVao));
}

void OpenGLMeshHandle::unbind()
{
    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLMeshHandle::addVertexBuffer(VertexBuffer* buffer, AttribType type)
{
    assert(buffer != nullptr);

    CHECK_ERROR(glBindVertexArray(mVao));
    buffer->bind();

    switch (type)
    {
    case AttribType::Vec2:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(glVertexAttribPointer(mVertexAttribIndex, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0));
        mVertexAttribIndex++;
        break;
    case AttribType::Vec3:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(glVertexAttribPointer(mVertexAttribIndex, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0));
        mVertexAttribIndex++;
        break;
    case AttribType::Vec4:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(glVertexAttribPointer(mVertexAttribIndex, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GL_FLOAT), 0));
        // TODO: This is used when the buffer is used in instancing. Pass bool to indicate when buffer is an instancing
        // buffer?
        CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, 1));
        mVertexAttribIndex++;
        break;
    case AttribType::Color32:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(
            glVertexAttribPointer(mVertexAttribIndex, 4, GL_UNSIGNED_INT, GL_FALSE, 4 * sizeof(GL_UNSIGNED_INT), 0));
        // TODO: This is used when the buffer is used in instancing. Pass bool to indicate when buffer is an instancing
        // buffer?
        CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, 1));
        mVertexAttribIndex++;
        break;
    case AttribType::Mat4:
        for (int i = 0; i < 4; i++)
        {
            CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
            CHECK_ERROR(glVertexAttribPointer(mVertexAttribIndex, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(sizeof(glm::vec4) * i)));
            // TODO: This is used when the buffer is used in instancing. Pass bool to indicate when buffer is an
            // instancing buffer?
            CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, 1));
            mVertexAttribIndex++;
        }
        break;
    }

    mBuffers.push_back(buffer);

    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLMeshHandle::addIndexBuffer(IndexBuffer *buffer)
{
    assert(buffer != nullptr);

    CHECK_ERROR(glBindVertexArray(mVao));
    buffer->bind();
    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLMeshHandle::drawLines(size_t vertexOffset, size_t vertexCount)
{
    CHECK_ERROR(glBindVertexArray(mVao));
    CHECK_ERROR(glDrawArrays(GL_LINES, (GLint)vertexOffset, (GLsizei)vertexCount));
    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLMeshHandle::draw(size_t vertexOffset, size_t vertexCount)
{
    CHECK_ERROR(glBindVertexArray(mVao));
    CHECK_ERROR(glDrawArrays(GL_TRIANGLES, (GLint)vertexOffset, (GLsizei)vertexCount));
    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLMeshHandle::drawIndexed(size_t indexOffset, size_t indexCount)
{
    CHECK_ERROR(glBindVertexArray(mVao));
    CHECK_ERROR(glDrawElements(GL_TRIANGLES, (GLsizei)indexCount, GL_UNSIGNED_INT, 0));
    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLMeshHandle::drawInstanced(size_t vertexOffset, size_t vertexCount, size_t instanceCount)
{
    CHECK_ERROR(glBindVertexArray(mVao));
    CHECK_ERROR(glDrawArraysInstanced(GL_TRIANGLES, (GLint)vertexOffset, (GLsizei)vertexCount, (GLsizei)instanceCount));
    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLMeshHandle::drawIndexedInstanced(size_t indexOffset, size_t indexCount, size_t instanceCount)
{
    CHECK_ERROR(glBindVertexArray(mVao));
    CHECK_ERROR(glDrawElementsInstanced(GL_TRIANGLES, (GLsizei)indexCount, GL_UNSIGNED_INT, 0, (GLsizei)instanceCount));
    CHECK_ERROR(glBindVertexArray(0));
}
