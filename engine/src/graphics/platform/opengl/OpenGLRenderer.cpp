#include <GL/glew.h>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <random>

#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/opengl/OpenGLRenderer.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"

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

    std::string version = (const char*)glGetString(GL_VERSION);
    std::string shader_version = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
    std::string vendor = (const char*)glGetString(GL_VENDOR);
    std::string renderer = (const char*)glGetString(GL_RENDERER);

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

void OpenGLRenderer::bindFramebuffer_impl(Framebuffer* fbo)
{
    if (fbo == nullptr)
    {
        CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));            
    }
    else
    {
        fbo->bind();
    }
}

void OpenGLRenderer::unbindFramebuffer_impl()
{
    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

void OpenGLRenderer::readColorAtPixel_impl(Framebuffer *fbo, int x, int y, Color32 *color)
{
    if (fbo != nullptr)
    {
        fbo->bind();
        CHECK_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT0));
        CHECK_ERROR(glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, color));
        fbo->unbind();
    }
}

void OpenGLRenderer::clearFrambufferColor_impl(const Color &color)
{
    CHECK_ERROR(glClearColor(color.mR, color.mG, color.mB, color.mA));
    CHECK_ERROR(glClear(GL_COLOR_BUFFER_BIT));
}

void OpenGLRenderer::clearFrambufferColor_impl(float r, float g, float b, float a)
{
    CHECK_ERROR(glClearColor(r, g, b, a));
    CHECK_ERROR(glClear(GL_COLOR_BUFFER_BIT));
}

void OpenGLRenderer::clearFramebufferDepth_impl(float depth)
{
    CHECK_ERROR(glClearDepth(depth));
    CHECK_ERROR(glClear(GL_DEPTH_BUFFER_BIT));
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

void OpenGLRenderer::draw_impl(const RenderObject &renderObject, GraphicsQuery &query)
{
    assert(renderObject.instanced == false);

    renderObject.meshHandle->draw(renderObject.start / 3, renderObject.size / 3);

    unsigned int count = renderObject.size / 3;

    query.mNumDrawCalls++;
    query.mVerts += count;
    query.mTris += count / 3;
}

void OpenGLRenderer::drawIndexed_impl(const RenderObject &renderObject, GraphicsQuery &query)
{
    assert(renderObject.instanced == false);

    renderObject.meshHandle->drawIndexed(renderObject.start, renderObject.size);

    unsigned int count = renderObject.size;

    query.mNumDrawCalls++;
    query.mVerts += count;
    query.mTris += count / 3;
}

void OpenGLRenderer::drawInstanced_impl(const RenderObject &renderObject, GraphicsQuery &query)
{
    assert(renderObject.instanced == true);

    renderObject.meshHandle->drawInstanced(renderObject.start / 3, renderObject.size / 3, renderObject.instanceCount);

    unsigned int count = renderObject.size / 3;

    query.mNumDrawCalls++;
    query.mVerts += count;
    query.mTris += count / 3;
}

void OpenGLRenderer::drawIndexedInstanced_impl(const RenderObject &renderObject, GraphicsQuery &query)
{
    assert(renderObject.instanced == true);

    renderObject.meshHandle->drawIndexedInstanced(renderObject.start, renderObject.size, renderObject.instanceCount);

    unsigned int count = renderObject.size;

    query.mNumDrawCalls++;
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

void OpenGLRenderer::createScreenQuad_impl(unsigned int *vao, unsigned int *vbo)
{
    float quadVertices[] = {
        // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
        // positions   // texCoords
        -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f, 0.0f,

        -1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  -1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  1.0f, 1.0f};

    CHECK_ERROR(glGenVertexArrays(1, vao));
    CHECK_ERROR(glGenBuffers(1, vbo));
    CHECK_ERROR(glBindVertexArray(*vao));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, *vbo));
    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW));
    CHECK_ERROR(glEnableVertexAttribArray(0));
    CHECK_ERROR(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0));
    CHECK_ERROR(glEnableVertexAttribArray(1));
    CHECK_ERROR(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float))));

    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLRenderer::renderScreenQuad_impl(unsigned int vao)
{
    CHECK_ERROR(glDisable(GL_DEPTH_TEST));
    CHECK_ERROR(glBindVertexArray(vao));
    CHECK_ERROR(glDrawArrays(GL_TRIANGLES, 0, 6));
    CHECK_ERROR(glBindVertexArray(0));
    CHECK_ERROR(glEnable(GL_DEPTH_TEST));
}