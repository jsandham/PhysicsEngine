#include <GL/glew.h>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <random>

#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"
#include "../../../../include/graphics/platform/opengl/OpenGLRenderer.h"

#include "../../../../include/graphics/InternalShaders.h"

using namespace PhysicsEngine;

struct Uniform
{
    GLsizei nameLength;
    GLint size;
    GLenum type;
    GLchar name[32];
};

struct Attribute
{
    GLsizei nameLength;
    GLint size;
    GLenum type;
    GLchar name[32];
};

void OpenGLRenderer::init_impl()
{
    mContext = OpenGLRenderContext::get();

    std::string version = (const char *)glGetString(GL_VERSION);
    std::string shader_version = (const char *)glGetString(GL_SHADING_LANGUAGE_VERSION);
    std::string vendor = (const char *)glGetString(GL_VENDOR);
    std::string renderer = (const char *)glGetString(GL_RENDERER);

    Log::warn(("Version: " + version + "\n").c_str());
    Log::warn(("Shader Version: " + shader_version + "\n").c_str());
    Log::warn(("Vendor: " + vendor + "\n").c_str());
    Log::warn(("Renderer: " + renderer + "\n").c_str());
}

void OpenGLRenderer::present_impl()
{
    mContext->present();
}

void OpenGLRenderer::turnVsyncOn_impl()
{
    mContext->turnVsyncOn();
}

void OpenGLRenderer::turnVsyncOff_impl()
{
    mContext->turnVsyncOff();
}

void OpenGLRenderer::bindBackBuffer_impl()
{
    mContext->bindBackBuffer();
}

void OpenGLRenderer::unbindBackBuffer_impl()
{
    mContext->unBindBackBuffer();
}

void OpenGLRenderer::clearBackBufferColor_impl(const Color &color)
{
    this->clearBackBufferColor_impl(color.mR, color.mG, color.mB, color.mA);
}

void OpenGLRenderer::clearBackBufferColor_impl(float r, float g, float b, float a)
{
    CHECK_ERROR(glClearColor(r, g, b, a));
    CHECK_ERROR(glClear(GL_COLOR_BUFFER_BIT));
}

void OpenGLRenderer::setViewport_impl(int x, int y, int width, int height)
{
    CHECK_ERROR(glViewport(x, y, width, height));
}

void OpenGLRenderer::turnOn_impl(Capability capability)
{
    switch (capability)
    {
    case Capability::Depth_Testing:
        CHECK_ERROR(glEnable(GL_DEPTH_TEST));
        break;
    case Capability::Blending:
        CHECK_ERROR(glEnable(GL_BLEND));
        break;
    case Capability::BackfaceCulling:
        CHECK_ERROR(glEnable(GL_CULL_FACE));
        break;
    case Capability::LineSmoothing:
        CHECK_ERROR(glEnable(GL_LINE_SMOOTH));
        break;
    }
}

void OpenGLRenderer::turnOff_impl(Capability capability)
{
    switch (capability)
    {
    case Capability::Depth_Testing:
        CHECK_ERROR(glDisable(GL_DEPTH_TEST));
        break;
    case Capability::Blending:
        CHECK_ERROR(glDisable(GL_BLEND));
        break;
    case Capability::BackfaceCulling:
        CHECK_ERROR(glDisable(GL_CULL_FACE));
        break;
    case Capability::LineSmoothing:
        CHECK_ERROR(glDisable(GL_LINE_SMOOTH));
        break;
    }
}

void OpenGLRenderer::setBlending_impl(BlendingFactor source, BlendingFactor dest)
{
    GLenum s = GL_ZERO;
    switch (source)
    {
    case BlendingFactor::ZERO:
        s = GL_ZERO;
        break;
    case BlendingFactor::ONE:
        s = GL_ONE;
        break;
    case BlendingFactor::SRC_ALPHA:
        s = GL_SRC_ALPHA;
        break;
    case BlendingFactor::ONE_MINUS_SRC_ALPHA:
        s = GL_ONE_MINUS_SRC_ALPHA;
        break;
    }

    GLenum d = GL_ZERO;
    switch (dest)
    {
    case BlendingFactor::ZERO:
        d = GL_ZERO;
        break;
    case BlendingFactor::ONE:
        d = GL_ONE;
        break;
    case BlendingFactor::SRC_ALPHA:
        d = GL_SRC_ALPHA;
        break;
    case BlendingFactor::ONE_MINUS_SRC_ALPHA:
        d = GL_ONE_MINUS_SRC_ALPHA;
        break;
    }

    CHECK_ERROR(glBlendFunc(s, d));
}

void OpenGLRenderer::draw_impl(MeshHandle *meshHandle, size_t vertexOffset, size_t vertexCount, GraphicsQuery &query)
{
    meshHandle->draw(vertexOffset, vertexCount);

    query.mNumDrawCalls++;
    query.mVerts += vertexCount;
    query.mTris += vertexCount / 3;
}

void OpenGLRenderer::drawIndexed_impl(MeshHandle *meshHandle, size_t indexOffset, size_t indexCount, GraphicsQuery &query)
{
    meshHandle->drawIndexed(indexOffset, indexCount);

    unsigned int count = indexCount;

    query.mNumDrawCalls++;
    query.mVerts += count;
    query.mTris += count / 3;
}

void OpenGLRenderer::drawInstanced_impl(MeshHandle *meshHandle, size_t vertexOffset, size_t vertexCount, size_t instanceCount, GraphicsQuery &query)
{
    meshHandle->drawInstanced(vertexOffset, vertexCount, instanceCount);

    query.mNumInstancedDrawCalls++;
    query.mVerts += vertexCount;
    query.mTris += vertexCount / 3;
}

void OpenGLRenderer::drawIndexedInstanced_impl(MeshHandle *meshHandle, size_t indexOffset, size_t indexCount, size_t instanceCount, GraphicsQuery &query)
{
    meshHandle->drawIndexedInstanced(indexOffset, indexCount, instanceCount);

    unsigned int count = indexCount;

    query.mNumInstancedDrawCalls++;
    query.mVerts += count;
    query.mTris += count / 3;
}

void OpenGLRenderer::beginQuery_impl(unsigned int queryId)
{
    CHECK_ERROR(glBeginQuery(GL_TIME_ELAPSED, queryId));
}

void OpenGLRenderer::endQuery_impl(unsigned int queryId, unsigned long long *elapsedTime)
{
    CHECK_ERROR(glEndQuery(GL_TIME_ELAPSED));
    CHECK_ERROR(glGetQueryObjectui64v(queryId, GL_QUERY_RESULT, elapsedTime));
}