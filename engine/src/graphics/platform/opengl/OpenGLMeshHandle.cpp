#include "../../../../include/graphics/platform/opengl/OpenGLMeshHandle.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"
#include "../../../../include/core/glm.h"

#include <GL/glew.h>
#include <assert.h>

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

void OpenGLMeshHandle::addVertexBuffer(VertexBuffer *buffer, std::string name, AttribType type, bool instanceBuffer)
{
    assert(buffer != nullptr);

    int divisor = instanceBuffer ? 1 : 0;

    CHECK_ERROR(glBindVertexArray(mVao));
    buffer->bind(mVertexAttribIndex);

    switch (type)
    {
    case AttribType::Int:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(glVertexAttribIPointer(mVertexAttribIndex, 1, GL_INT, sizeof(GL_INT), 0));
        CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, divisor));
        mVertexAttribIndex++;
        break;
    case AttribType::Float:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(glVertexAttribPointer(mVertexAttribIndex, 1, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT), 0));
        CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, divisor));
        mVertexAttribIndex++;
        break;
    case AttribType::Vec2:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(glVertexAttribPointer(mVertexAttribIndex, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0));
        CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, divisor));
        mVertexAttribIndex++;
        break;
    case AttribType::Vec3:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(glVertexAttribPointer(mVertexAttribIndex, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0));
        CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, divisor));
        mVertexAttribIndex++;
        break;
    case AttribType::Vec4:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(glVertexAttribPointer(mVertexAttribIndex, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GL_FLOAT), 0));
        CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, divisor));
        mVertexAttribIndex++;
        break;
    case AttribType::IVec2:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(glVertexAttribIPointer(mVertexAttribIndex, 2, GL_INT, 2 * sizeof(GL_INT), 0));
        CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, divisor));
        mVertexAttribIndex++;
        break;
    case AttribType::IVec3:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(glVertexAttribIPointer(mVertexAttribIndex, 3, GL_INT, 3 * sizeof(GL_INT), 0));
        CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, divisor));
        mVertexAttribIndex++;
        break;
    case AttribType::IVec4:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(glVertexAttribIPointer(mVertexAttribIndex, 4, GL_INT, 4 * sizeof(GL_INT), 0));
        CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, divisor));
        mVertexAttribIndex++;
        break;
    case AttribType::UVec2:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(glVertexAttribIPointer(mVertexAttribIndex, 2, GL_UNSIGNED_INT, 2 * sizeof(GL_UNSIGNED_INT), 0));
        CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, divisor));
        mVertexAttribIndex++;
        break;
    case AttribType::UVec3:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(glVertexAttribIPointer(mVertexAttribIndex, 3, GL_UNSIGNED_INT, 3 * sizeof(GL_UNSIGNED_INT), 0));
        CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, divisor));
        mVertexAttribIndex++;
        break;
    case AttribType::UVec4:
        CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
        CHECK_ERROR(glVertexAttribIPointer(mVertexAttribIndex, 4, GL_UNSIGNED_INT, 4 * sizeof(GL_UNSIGNED_INT), 0));
        CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, divisor));
        mVertexAttribIndex++;
        break;
    case AttribType::Mat4:
        for (int i = 0; i < 4; i++)
        {
            CHECK_ERROR(glEnableVertexAttribArray(mVertexAttribIndex));
            CHECK_ERROR(glVertexAttribPointer(mVertexAttribIndex, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4),
                                              (void *)(sizeof(glm::vec4) * i)));
            CHECK_ERROR(glVertexAttribDivisor(mVertexAttribIndex, divisor));
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
    CHECK_ERROR(glDrawElements(GL_TRIANGLES, (GLsizei)indexCount, GL_UNSIGNED_INT, (void*)(indexOffset * sizeof(unsigned int))));
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
    CHECK_ERROR(glDrawElementsInstanced(GL_TRIANGLES, (GLsizei)indexCount, GL_UNSIGNED_INT, (void*)(indexOffset * sizeof(unsigned int)), (GLsizei)instanceCount));
    CHECK_ERROR(glBindVertexArray(0));
}
