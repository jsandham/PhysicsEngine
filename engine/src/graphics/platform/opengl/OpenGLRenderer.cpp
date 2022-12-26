#include <GL/glew.h>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <random>

#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/opengl/OpenGLRenderer.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"
#include "GLSL/glsl_shaders.h"

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
        CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, *reinterpret_cast<unsigned int*>(fbo->getHandle())));
    }
}

void OpenGLRenderer::unbindFramebuffer_impl()
{
    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
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

void OpenGLRenderer::beginQuery_impl(unsigned int queryId)
{
    CHECK_ERROR(glBeginQuery(GL_TIME_ELAPSED, queryId));
}

void OpenGLRenderer::endQuery_impl(unsigned int queryId, unsigned long long *elapsedTime)
{
    CHECK_ERROR(glEndQuery(GL_TIME_ELAPSED));
    CHECK_ERROR(glGetQueryObjectui64v(queryId, GL_QUERY_RESULT, elapsedTime));
}

//void OpenGLRenderer::createGlobalCameraUniforms_impl(CameraUniform &uniform)
//{
//    CHECK_ERROR(glGenBuffers(1, &uniform.mBuffer));
//    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, uniform.mBuffer));
//    CHECK_ERROR(glBufferData(GL_UNIFORM_BUFFER, 204, NULL, GL_DYNAMIC_DRAW));
//    CHECK_ERROR(glBindBufferRange(GL_UNIFORM_BUFFER, 0, uniform.mBuffer, 0, 204));
//    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, 0));
//}
//
//void OpenGLRenderer::createGlobalLightUniforms_impl(LightUniform &uniform)
//{
//    CHECK_ERROR(glGenBuffers(1, &uniform.mBuffer));
//    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, uniform.mBuffer));
//    CHECK_ERROR(glBufferData(GL_UNIFORM_BUFFER, 824, NULL, GL_DYNAMIC_DRAW));
//    CHECK_ERROR(glBindBufferRange(GL_UNIFORM_BUFFER, 1, uniform.mBuffer, 0, 824));
//    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, 0));
//}
//
//void OpenGLRenderer::setGlobalCameraUniforms_impl(const CameraUniform &uniform)
//{
//    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, uniform.mBuffer));
//    CHECK_ERROR(glBindBufferRange(GL_UNIFORM_BUFFER, 0, uniform.mBuffer, 0, 204));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 0, 64, glm::value_ptr(uniform.mProjection)));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 64, 64, glm::value_ptr(uniform.mView)));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 128, 64, glm::value_ptr(uniform.mViewProjection)));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 192, 12, glm::value_ptr(uniform.mCameraPos)));
//    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, 0));
//}
//
//void OpenGLRenderer::setGlobalLightUniforms_impl(const LightUniform &uniform)
//{
//    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, uniform.mBuffer));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 0, 320, &uniform.mLightProjection[0]));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 320, 320, &uniform.mLightView[0]));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 640, 12, glm::value_ptr(uniform.mPosition)));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 656, 12, glm::value_ptr(uniform.mDirection)));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 672, 16, glm::value_ptr(uniform.mColor)));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 688, 4, &uniform.mCascadeEnds[0]));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 704, 4, &uniform.mCascadeEnds[1]));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 720, 4, &uniform.mCascadeEnds[2]));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 736, 4, &uniform.mCascadeEnds[3]));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 752, 4, &uniform.mCascadeEnds[4]));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 768, 4, &uniform.mIntensity));
//
//    float spotAngle = glm::cos(glm::radians(uniform.mSpotAngle));
//    float innerSpotAngle = glm::cos(glm::radians(uniform.mInnerSpotAngle));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 772, 4, &(spotAngle)));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 776, 4, &(innerSpotAngle)));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 780, 4, &(uniform.mShadowNearPlane)));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 784, 4, &(uniform.mShadowFarPlane)));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 788, 4, &(uniform.mShadowBias)));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 792, 4, &(uniform.mShadowRadius)));
//    CHECK_ERROR(glBufferSubData(GL_UNIFORM_BUFFER, 796, 4, &(uniform.mShadowStrength)));
//    CHECK_ERROR(glBindBuffer(GL_UNIFORM_BUFFER, 0));
//}

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

//void OpenGLRenderer::createFramebuffer_impl(int width, int height, unsigned int *fbo, unsigned int *color)
//{
//    // generate fbo (color + depth)
//    CHECK_ERROR(glGenFramebuffers(1, fbo));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, *fbo));
//
//    CHECK_ERROR(glGenTextures(1, color));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *color));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *color, 0));
//
//    // - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
//    unsigned int mainAttachments[1] = {GL_COLOR_ATTACHMENT0};
//    CHECK_ERROR(glDrawBuffers(1, mainAttachments));
//
//    checkFrambufferError(std::to_string(__LINE__), std::string(__FILE__));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//}
//
//void OpenGLRenderer::createFramebuffer_impl(int width, int height, unsigned int *fbo, unsigned int *color,
//                                               unsigned int *depth)
//{
//    // generate fbo (color + depth)
//    CHECK_ERROR(glGenFramebuffers(1, fbo));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, *fbo));
//
//    CHECK_ERROR(glGenTextures(1, color));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *color));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//
//    CHECK_ERROR(glGenTextures(1, depth));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *depth));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *color, 0));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, *depth, 0));
//
//    // - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
//    unsigned int mainAttachments[1] = {GL_COLOR_ATTACHMENT0};
//    CHECK_ERROR(glDrawBuffers(1, mainAttachments));
//
//    checkFrambufferError(std::to_string(__LINE__), std::string(__FILE__));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//}
//
//void OpenGLRenderer::destroyFramebuffer_impl(unsigned int *fbo, unsigned int *color, unsigned int *depth)
//{
//    // detach textures from their framebuffer
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, *fbo));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//
//    // delete frambuffer
//    CHECK_ERROR(glDeleteFramebuffers(1, fbo));
//
//    // delete textures
//    CHECK_ERROR(glDeleteTextures(1, color));
//    CHECK_ERROR(glDeleteTextures(1, depth));
//}

void OpenGLRenderer::bindVertexArray_impl(unsigned int vao)
{
    CHECK_ERROR(glBindVertexArray(vao));
}

void OpenGLRenderer::unbindVertexArray_impl()
{
    CHECK_ERROR(glBindVertexArray(0));
}

//void OpenGLRenderer::createTargets_impl(CameraTargets *targets, Viewport viewport, glm::vec3 *ssaoSamples,
//                                           unsigned int *queryId0,
//                             unsigned int *queryId1)
//{
//    // generate timing queries
//    CHECK_ERROR(glGenQueries(1, queryId0));
//    CHECK_ERROR(glGenQueries(1, queryId1));
//
//    // dummy query to prevent OpenGL errors from popping out
//    // see https://www.lighthouse3d.com/tutorials/opengl-timer-query/
//    // CHECK_ERROR(glQueryCounter(*queryId1, GL_TIMESTAMP));
//
//    // generate main camera fbo (color + depth)
//    CHECK_ERROR(glGenFramebuffers(1, &(targets->mMainFBO)));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mMainFBO));
//
//    CHECK_ERROR(glGenTextures(1, &(targets->mColorTex)));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mColorTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//
//    CHECK_ERROR(glGenTextures(1, &(targets->mDepthTex)));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mDepthTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1920, 1080, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, targets->mColorTex, 0));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, targets->mDepthTex, 0));
//
//    // - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
//    unsigned int mainAttachments[1] = {GL_COLOR_ATTACHMENT0};
//    CHECK_ERROR(glDrawBuffers(1, mainAttachments));
//
//    checkFrambufferError(std::to_string(__LINE__), std::string(__FILE__));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//
//    // generate color picking fbo (color + depth)
//    CHECK_ERROR(glGenFramebuffers(1, &(targets->mColorPickingFBO)));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mColorPickingFBO));
//
//    CHECK_ERROR(glGenTextures(1, &(targets->mColorPickingTex)));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mColorPickingTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//
//    CHECK_ERROR(glGenTextures(1, &(targets->mColorPickingDepthTex)));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mColorPickingDepthTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1920, 1080, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, targets->mColorPickingTex, 0));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, targets->mColorPickingDepthTex, 0));
//
//    // - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
//    unsigned int colorPickingAttachments[1] = {GL_COLOR_ATTACHMENT0};
//    CHECK_ERROR(glDrawBuffers(1, colorPickingAttachments));
//
//    checkFrambufferError(std::to_string(__LINE__), std::string(__FILE__));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//
//    // generate geometry fbo
//    CHECK_ERROR(glGenFramebuffers(1, &(targets->mGeometryFBO)));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mGeometryFBO));
//
//    CHECK_ERROR(glGenTextures(1, &(targets->mPositionTex)));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mPositionTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//
//    CHECK_ERROR(glGenTextures(1, &(targets->mNormalTex)));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mNormalTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//
//    CHECK_ERROR(glGenTextures(1, &(targets->mAlbedoSpecTex)));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mAlbedoSpecTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, targets->mPositionTex, 0));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, targets->mNormalTex, 0));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, targets->mAlbedoSpecTex, 0));
//
//    unsigned int geometryAttachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
//    CHECK_ERROR(glDrawBuffers(3, geometryAttachments));
//
//    // create and attach depth buffer (renderbuffer)
//    unsigned int rboDepth;
//    CHECK_ERROR(glGenRenderbuffers(1, &rboDepth));
//    CHECK_ERROR(glBindRenderbuffer(GL_RENDERBUFFER, rboDepth));
//    CHECK_ERROR(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1920, 1080));
//    CHECK_ERROR(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth));
//
//    checkFrambufferError(std::to_string(__LINE__), std::string(__FILE__));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//
//    // generate ssao fbo
//    CHECK_ERROR(glGenFramebuffers(1, &(targets->mSsaoFBO)));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mSsaoFBO));
//
//    CHECK_ERROR(glGenTextures(1, &(targets->mSsaoColorTex)));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mSsaoColorTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, targets->mSsaoColorTex, 0));
//
//    unsigned int ssaoAttachments[1] = {GL_COLOR_ATTACHMENT0};
//    CHECK_ERROR(glDrawBuffers(1, ssaoAttachments));
//
//    checkFrambufferError(std::to_string(__LINE__), std::string(__FILE__));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//
//    auto lerp = [](float a, float b, float t) { return a + t * (b - a); };
//
//    // generate noise texture for use in ssao
//    std::uniform_real_distribution<GLfloat> distribution(0.0, 1.0);
//    std::default_random_engine generator;
//    for (unsigned int j = 0; j < 64; ++j)
//    {
//        float x = distribution(generator) * 2.0f - 1.0f;
//        float y = distribution(generator) * 2.0f - 1.0f;
//        float z = distribution(generator);
//        float radius = distribution(generator);
//
//        glm::vec3 sample(x, y, z);
//        sample = radius * glm::normalize(sample);
//        float scale = float(j) / 64.0f;
//
//        // scale samples s.t. they're more aligned to center of kernel
//        scale = lerp(0.1f, 1.0f, scale * scale);
//        sample *= scale;
//
//        ssaoSamples[j] = sample;
//    }
//
//    glm::vec3 ssaoNoise[16];
//    for (int j = 0; j < 16; j++)
//    {
//        // rotate around z-axis (in tangent space)
//        glm::vec3 noise(distribution(generator) * 2.0f - 1.0f, distribution(generator) * 2.0f - 1.0f, 0.0f);
//        ssaoNoise[j] = noise;
//    }
//
//    CHECK_ERROR(glGenTextures(1, &(targets->mSsaoNoiseTex)));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mSsaoNoiseTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssaoNoise[0]));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//}
//
//void OpenGLRenderer::destroyTargets_impl(CameraTargets *targets, unsigned int *queryId0, unsigned int *queryId1)
//{
//    // detach textures from their framebuffer
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mMainFBO));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mColorPickingFBO));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mGeometryFBO));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, 0, 0));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mSsaoFBO));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//
//    // delete frambuffers
//    CHECK_ERROR(glDeleteFramebuffers(1, &(targets->mMainFBO)));
//    CHECK_ERROR(glDeleteFramebuffers(1, &(targets->mColorPickingFBO)));
//    CHECK_ERROR(glDeleteFramebuffers(1, &(targets->mGeometryFBO)));
//    CHECK_ERROR(glDeleteFramebuffers(1, &(targets->mSsaoFBO)));
//
//    // delete textures
//    CHECK_ERROR(glDeleteTextures(1, &(targets->mColorTex)));
//    CHECK_ERROR(glDeleteTextures(1, &(targets->mDepthTex)));
//    CHECK_ERROR(glDeleteTextures(1, &(targets->mColorPickingTex)));
//    CHECK_ERROR(glDeleteTextures(1, &(targets->mColorPickingDepthTex)));
//    CHECK_ERROR(glDeleteTextures(1, &(targets->mPositionTex)));
//    CHECK_ERROR(glDeleteTextures(1, &(targets->mNormalTex)));
//    CHECK_ERROR(glDeleteTextures(1, &(targets->mAlbedoSpecTex)));
//    CHECK_ERROR(glDeleteTextures(1, &(targets->mSsaoColorTex)));
//    CHECK_ERROR(glDeleteTextures(1, &(targets->mSsaoNoiseTex)));
//
//    // delete timing query
//    CHECK_ERROR(glDeleteQueries(1, queryId0));
//    CHECK_ERROR(glDeleteQueries(1, queryId1));
//}
//
//void OpenGLRenderer::resizeTargets_impl(CameraTargets *targets, Viewport viewport, bool *viewportChanged)
//{
//    /*int width = camera->getViewport().mWidth;
//    int height = camera->getViewport().mHeight;
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *colorTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *depthTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *colorPickingTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *colorPickingDepthTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//
//    if (camera->mSSAO == CameraSSAO::SSAO_On) {
//        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *positionTex));
//        CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL));
//        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//
//        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *normalTex));
//        CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL));
//        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//
//        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *albedoSpecTex));
//        CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
//        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//
//        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *ssaoColorTex));
//        CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL));
//        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//    }
//
//    *viewportChanged = false;*/
//}

void OpenGLRenderer::readColorAtPixel_impl(const unsigned int *fbo, int x, int y, Color32 *color)
{
    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, *fbo));
    CHECK_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT0));
    CHECK_ERROR(glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, color));
    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

//void OpenGLRenderer::createTargets_impl(LightTargets *targets, ShadowMapResolution resolution)
//{
//    GLsizei res = 512;
//    switch (resolution)
//    {
//    case ShadowMapResolution::Low512x512:
//        res = 512;
//        break;
//    case ShadowMapResolution::Medium1024x1024:
//        res = 1024;
//        break;
//    case ShadowMapResolution::High2048x2048:
//        res = 2048;
//        break;
//    case ShadowMapResolution::VeryHigh4096x4096:
//        res = 4096;
//        break;
//    default:
//        res = 1024;
//        break;
//    }
//
//    // generate shadow map fbos
//    // create directional light cascade shadow map fbo
//    CHECK_ERROR(glGenFramebuffers(5, &(targets->mShadowCascadeFBO[0])));
//    CHECK_ERROR(glGenTextures(5, &(targets->mShadowCascadeDepthTex[0])));
//
//    for (int i = 0; i < 5; i++)
//    {
//        CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mShadowCascadeFBO[i]));
//        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mShadowCascadeDepthTex[i]));
//
//        CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));
//        CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//        CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//        CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
//        CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
//
//        CHECK_ERROR(glDrawBuffer(GL_NONE));
//        CHECK_ERROR(glReadBuffer(GL_NONE));
//
//        CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, targets->mShadowCascadeDepthTex[i],
//                               0));
//
//        checkFrambufferError(std::to_string(__LINE__), std::string(__FILE__));
//
//        CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//    }
//
//    // create spotlight shadow map fbo
//    CHECK_ERROR(glGenFramebuffers(1, &(targets->mShadowSpotlightFBO)));
//    CHECK_ERROR(glGenTextures(1, &(targets->mShadowSpotlightDepthTex)));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mShadowSpotlightFBO));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mShadowSpotlightDepthTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//    // CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER));
//    // CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
//
//    CHECK_ERROR(glDrawBuffer(GL_NONE));
//    CHECK_ERROR(glReadBuffer(GL_NONE));
//
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, targets->mShadowSpotlightDepthTex, 0));
//
//    checkFrambufferError(std::to_string(__LINE__), std::string(__FILE__));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//
//    // create pointlight shadow cubemap fbo
//    CHECK_ERROR(glGenFramebuffers(1, &(targets->mShadowCubemapFBO)));
//    CHECK_ERROR(glGenTextures(1, &(targets->mShadowCubemapDepthTex)));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mShadowCubemapFBO));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, targets->mShadowCubemapDepthTex));
//
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 0, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
//                 NULL));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 1, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
//                 NULL));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 2, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
//                 NULL));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 3, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
//                 NULL));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 4, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
//                 NULL));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 5, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
//                 NULL));
//
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
//
//    CHECK_ERROR(glDrawBuffer(GL_NONE));
//    CHECK_ERROR(glReadBuffer(GL_NONE));
//
//    CHECK_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, targets->mShadowCubemapDepthTex, 0));
//
//    checkFrambufferError(std::to_string(__LINE__), std::string(__FILE__));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//}
//
//void OpenGLRenderer::destroyTargets_impl(LightTargets *targets)
//{
//    // detach textures from their framebuffer
//    for (int i = 0; i < 5; i++)
//    {
//        CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mShadowCascadeFBO[i]));
//        CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0));
//        CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//    }
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mShadowSpotlightFBO));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mShadowCubemapFBO));
//    CHECK_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 0, 0));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//
//    // delete frambuffers
//    for (int i = 0; i < 5; i++)
//    {
//        CHECK_ERROR(glDeleteFramebuffers(1, &(targets->mShadowCascadeFBO[i])));
//    }
//    CHECK_ERROR(glDeleteFramebuffers(1, &(targets->mShadowSpotlightFBO)));
//    CHECK_ERROR(glDeleteFramebuffers(1, &(targets->mShadowCubemapFBO)));
//
//    // delete textures
//    for (int i = 0; i < 5; i++)
//    {
//        CHECK_ERROR(glDeleteTextures(1, &(targets->mShadowCascadeDepthTex[i])));
//    }
//    CHECK_ERROR(glDeleteTextures(1, &(targets->mShadowSpotlightDepthTex)));
//    CHECK_ERROR(glDeleteTextures(1, &(targets->mShadowCubemapDepthTex)));
//}
//
//void OpenGLRenderer::resizeTargets_impl(LightTargets *targets, ShadowMapResolution resolution)
//{
//    GLsizei res = 512;
//    switch (resolution)
//    {
//    case ShadowMapResolution::Low512x512:
//        res = 512;
//        break;
//    case ShadowMapResolution::Medium1024x1024:
//        res = 1024;
//        break;
//    case ShadowMapResolution::High2048x2048:
//        res = 2048;
//        break;
//    case ShadowMapResolution::VeryHigh4096x4096:
//        res = 4096;
//        break;
//    default:
//        res = 1024;
//        break;
//    }
//
//    // If resolution not actually changed, return
//    /*if (res == static_cast<GLsizei>(resolution)) {
//        *resolutionChanged = false;
//        return;
//    }*/
//
//    for (int i = 0; i < 5; i++)
//    {
//        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mShadowCascadeDepthTex[i]));
//        CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));
//        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//    }
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mShadowSpotlightDepthTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, targets->mShadowCubemapDepthTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 0, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
//                 NULL));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 1, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
//                 NULL));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 2, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
//                 NULL));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 3, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
//                 NULL));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 4, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
//                 NULL));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 5, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
//                 NULL));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//}

//void OpenGLRenderer::createTexture2D_impl(TextureFormat format, TextureWrapMode wrapMode,
//                                             TextureFilterMode filterMode, int width,
//                               int height, const std::vector<unsigned char> &data, TextureHandle*tex /*unsigned int* tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    CHECK_ERROR(glGenTextures(1, &opengltex->mHandle /*tex*/));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, opengltex->mHandle /* *tex*/));
//
//    GLenum openglFormat = getTextureFormat(format);
//    GLint openglWrapMode = getTextureWrapMode(wrapMode);
//    GLint openglFilterMode = getTextureFilterMode(filterMode);
//
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, data.data()));
//
//    CHECK_ERROR(glGenerateMipmap(GL_TEXTURE_2D));
//
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, openglFilterMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
//                    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, openglWrapMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, openglWrapMode));
//
//    // clamp the requested anisotropic filtering level to what is available and set it
//    CHECK_ERROR(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 1.0f));
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//}
//
//void OpenGLRenderer::destroyTexture2D_impl(TextureHandle*tex /*unsigned int* tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    CHECK_ERROR(glDeleteTextures(1, &opengltex->mHandle /*tex*/));
//}
//
//void OpenGLRenderer::updateTexture2D_impl(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel,
//                                             TextureHandle*tex /*unsigned int tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    GLint openglWrapMode = getTextureWrapMode(wrapMode);
//    GLint openglFilterMode = getTextureFilterMode(filterMode);
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, opengltex->mHandle /*tex*/));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, openglFilterMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
//                    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, openglWrapMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, openglWrapMode));
//
//    // Determine how many levels of anisotropic filtering are available
//    float aniso = 0.0f;
//    CHECK_ERROR(glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso));
//
//    // clamp the requested anisotropic filtering level to what is available and set it
//    CHECK_ERROR(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, glm::clamp((float)anisoLevel, 1.0f, aniso)));
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//}
//
//void OpenGLRenderer::readPixelsTexture2D_impl(TextureFormat format, int width, int height, int numChannels,
//                                   std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, opengltex->mHandle /*tex*/));
//
//    GLenum openglFormat = getTextureFormat(format);
//
//    CHECK_ERROR(glGetTextureImage(opengltex->mHandle /*tex*/, 0, openglFormat, GL_UNSIGNED_BYTE, width * height * numChannels, &data[0]));
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//}
//
//void OpenGLRenderer::writePixelsTexture2D_impl(TextureFormat format, int width, int height,
//                                                  const std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, opengltex->mHandle /*tex*/));
//
//    GLenum openglFormat = getTextureFormat(format);
//
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, data.data()));
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//}
//
//void OpenGLRenderer::createTexture3D_impl(TextureFormat format, TextureWrapMode wrapMode,
//                                             TextureFilterMode filterMode, int width,
//                               int height, int depth, const std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int* tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    CHECK_ERROR(glGenTextures(1, &opengltex->mHandle /*tex*/));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_3D, opengltex->mHandle /**tex*/));
//
//    GLenum openglFormat = getTextureFormat(format);
//    GLint openglWrapMode = getTextureWrapMode(wrapMode);
//    GLint openglFilterMode = getTextureFilterMode(filterMode);
//
//    CHECK_ERROR(glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE, &data[0]));
//
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, openglFilterMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER,
//                    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, openglWrapMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, openglWrapMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, openglWrapMode));
//
//    // clamp the requested anisotropic filtering level to what is available and set it
//    CHECK_ERROR(glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 1.0f));
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_3D, 0));
//}
//
//void OpenGLRenderer::destroyTexture3D_impl(TextureHandle* tex /*unsigned int* tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    CHECK_ERROR(glDeleteTextures(1, &opengltex->mHandle /*tex*/));
//}
//
//void OpenGLRenderer::updateTexture3D_impl(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel,
//    TextureHandle* tex /*unsigned int tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    GLint openglWrapMode = getTextureWrapMode(wrapMode);
//    GLint openglFilterMode = getTextureFilterMode(filterMode);
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_3D,opengltex->mHandle /*tex*/));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, openglFilterMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER,
//                    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, openglWrapMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, openglWrapMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, openglWrapMode));
//
//    // Determine how many levels of anisotropic filtering are available
//    float aniso = 0.0f;
//    CHECK_ERROR(glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso));
//
//    // clamp the requested anisotropic filtering level to what is available and set it
//    CHECK_ERROR(glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAX_ANISOTROPY_EXT, glm::clamp((float)anisoLevel, 1.0f, aniso)));
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_3D, 0));
//}
//
//void OpenGLRenderer::readPixelsTexture3D_impl(TextureFormat format, int width, int height, int depth,
//                                                 int numChannels,
//                                   std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_3D, opengltex->mHandle /*tex*/));
//
//    GLenum openglFormat = getTextureFormat(format);
//
//    CHECK_ERROR(glGetTextureImage(opengltex->mHandle /*tex*/, 0, openglFormat, GL_UNSIGNED_BYTE, width * height * depth * numChannels, &data[0]));
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_3D, 0));
//}
//
//void OpenGLRenderer::writePixelsTexture3D_impl(TextureFormat format, int width, int height, int depth,
//                                    const std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_3D, opengltex->mHandle /*tex*/));
//
//    GLenum openglFormat = getTextureFormat(format);
//
//    CHECK_ERROR(glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE, &data[0]));
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_3D, 0));
//}
//
//void OpenGLRenderer::createCubemap_impl(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode,
//                                           int width,
//                             const std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int* tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    CHECK_ERROR(glGenTextures(1, &opengltex->mHandle /*tex*/));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, opengltex->mHandle /* *tex*/));
//
//    GLenum openglFormat = getTextureFormat(format);
//    GLint openglWrapMode = getTextureWrapMode(wrapMode);
//    GLint openglFilterMode = getTextureFilterMode(filterMode);
//
//    for (unsigned int i = 0; i < 6; i++)
//    {
//        CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, width, width, 0, openglFormat,
//                     GL_UNSIGNED_BYTE, data.data()));
//    }
//
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, openglFilterMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER,
//                    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, openglWrapMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, openglWrapMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, openglWrapMode));
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, 0));
//}
//
//void OpenGLRenderer::destroyCubemap_impl(TextureHandle* tex /*unsigned int* tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    CHECK_ERROR(glDeleteTextures(1, &opengltex->mHandle /*tex*/));
//}
//
//void OpenGLRenderer::updateCubemap_impl(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel,
//    TextureHandle* tex /*unsigned int tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    GLint openglWrapMode = getTextureWrapMode(wrapMode);
//    GLint openglFilterMode = getTextureFilterMode(filterMode);
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, opengltex->mHandle /*tex*/));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, openglFilterMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER,
//                    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, openglWrapMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, openglWrapMode));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, openglWrapMode));
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, 0));
//}
//
//void OpenGLRenderer::readPixelsCubemap_impl(TextureFormat format, int width, int numChannels,
//                                               std::vector<unsigned char> &data,
//    TextureHandle* tex /*unsigned int tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, opengltex->mHandle /*tex*/));
//
//    GLenum openglFormat = getTextureFormat(format);
//
//    for (unsigned int i = 0; i < 6; i++)
//    {
//        CHECK_ERROR(glGetTexImage(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, GL_UNSIGNED_BYTE,
//                      &data[i * width * width * numChannels]));
//    }
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, 0));
//}
//
//void OpenGLRenderer::writePixelsCubemap_impl(TextureFormat format, int width, const std::vector<unsigned char> &data,
//    TextureHandle* tex /*unsigned int tex*/)
//{
//    OpenGLTextureHandle* opengltex = static_cast<OpenGLTextureHandle*>(tex);
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, opengltex->mHandle /*tex*/));
//
//    GLenum openglFormat = getTextureFormat(format);
//
//    for (unsigned int i = 0; i < 6; i++)
//    {
//        CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, width, width, 0, openglFormat,
//                     GL_UNSIGNED_BYTE, data.data()));
//    }
//
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, 0));
//}

//void OpenGLRenderer::createRenderTextureTargets_impl(RenderTextureTargets *targets, TextureFormat format,
//                                                        TextureWrapMode wrapMode,
//                                          TextureFilterMode filterMode, int width, int height)
//{
//    // generate fbo (color + depth)
//    CHECK_ERROR(glGenFramebuffers(1, &(targets->mMainFBO)));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mMainFBO));
//
//    CHECK_ERROR(glGenTextures(1, &(targets->mColorTex)));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mColorTex));
//
//    GLenum openglFormat = getTextureFormat(format);
//    // GLint openglWrapMode = getTextureWrapMode(wrapMode);
//    // GLint openglFilterMode = getTextureFilterMode(filterMode);
//
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, NULL));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//
//    // CHECK_ERROR(glGenerateMipmap(GL_TEXTURE_2D);
//
//    // CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, openglFilterMode));
//    // CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
//    //     openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode));
//    // CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, openglWrapMode));
//    // CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, openglWrapMode));
//
//    // float aniso = 0.0f;
//    // CHECK_ERROR(glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso));
//
//    CHECK_ERROR(glGenTextures(1, &(targets->mDepthTex)));
//    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, targets->mDepthTex));
//    CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
//    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
//    // CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, openglFilterMode));
//    // CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
//    //     openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode));
//    // CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, openglWrapMode));
//    // CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, openglWrapMode));
//
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, targets->mColorTex, 0));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, targets->mDepthTex, 0));
//
//    // - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
//    unsigned int mainAttachments[1] = {GL_COLOR_ATTACHMENT0};
//    CHECK_ERROR(glDrawBuffers(1, mainAttachments));
//
//    checkFrambufferError(std::to_string(__LINE__), std::string(__FILE__));
//
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//}
//
//void OpenGLRenderer::destroyRenderTextureTargets_impl(RenderTextureTargets *targets)
//{
//    // detach textures from their framebuffer
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, targets->mMainFBO));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0));
//    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0));
//    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
//
//    // delete frambuffers
//    CHECK_ERROR(glDeleteFramebuffers(1, &(targets->mMainFBO)));
//
//    // delete textures
//    CHECK_ERROR(glDeleteTextures(1, &(targets->mColorTex)));
//    CHECK_ERROR(glDeleteTextures(1, &(targets->mDepthTex)));
//}

void OpenGLRenderer::createTerrainChunk_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
                                  const std::vector<float> &texCoords, int vertexCount, unsigned int *vao,
                                  unsigned int *vbo0, unsigned int *vbo1, unsigned int *vbo2)
{
    CHECK_ERROR(glGenVertexArrays(1, vao));
    CHECK_ERROR(glBindVertexArray(*vao));
    CHECK_ERROR(glGenBuffers(1, vbo0)); // vertex vbo
    CHECK_ERROR(glGenBuffers(1, vbo1)); // normals vbo
    CHECK_ERROR(glGenBuffers(1, vbo2)); // texcoords vbo

    CHECK_ERROR(glBindVertexArray(*vao));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, *vbo0));
    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, 81 * 3 * vertexCount * sizeof(float), NULL, GL_DYNAMIC_DRAW));
    CHECK_ERROR(glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data()));
    CHECK_ERROR(glEnableVertexAttribArray(0));
    CHECK_ERROR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0));

    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, *vbo1));
    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, 81 * 3 * vertexCount * sizeof(float), NULL, GL_DYNAMIC_DRAW));
    CHECK_ERROR(glBufferSubData(GL_ARRAY_BUFFER, 0, normals.size() * sizeof(float), normals.data()));
    CHECK_ERROR(glEnableVertexAttribArray(1));
    CHECK_ERROR(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0));

    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, *vbo2));
    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, 81 * 2 * vertexCount * sizeof(float), NULL, GL_DYNAMIC_DRAW));
    CHECK_ERROR(glBufferSubData(GL_ARRAY_BUFFER, 0, texCoords.size() * sizeof(float), texCoords.data()));
    CHECK_ERROR(glEnableVertexAttribArray(2));
    CHECK_ERROR(glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0));

    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLRenderer::destroyTerrainChunk_impl(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1,
                                                 unsigned int *vbo2)
{
    CHECK_ERROR(glDeleteBuffers(1, vbo0));
    CHECK_ERROR(glDeleteBuffers(1, vbo1));
    CHECK_ERROR(glDeleteBuffers(1, vbo2));

    CHECK_ERROR(glDeleteVertexArrays(1, vao));
}

void OpenGLRenderer::updateTerrainChunk_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
                                  unsigned int vbo0, unsigned int vbo1)
{
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo0));
    CHECK_ERROR(glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data()));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));

    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo1));
    CHECK_ERROR(glBufferSubData(GL_ARRAY_BUFFER, 0, normals.size() * sizeof(float), normals.data()));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void OpenGLRenderer::updateTerrainChunk_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
                                  const std::vector<float> &texCoords, unsigned int vbo0, unsigned int vbo1,
                                  unsigned int vbo2)
{
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo0));
    CHECK_ERROR(glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data()));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));

    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo1));
    CHECK_ERROR(glBufferSubData(GL_ARRAY_BUFFER, 0, normals.size() * sizeof(float), normals.data()));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));

    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo2));
    CHECK_ERROR(glBufferSubData(GL_ARRAY_BUFFER, 0, texCoords.size() * sizeof(float), texCoords.data()));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

//void OpenGLRenderer::createMesh_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
//                          const std::vector<float> &texCoords, unsigned int *vao, VertexBuffer*vbo0,
//    VertexBuffer*vbo1, VertexBuffer*vbo2, VertexBuffer*model_vbo, VertexBuffer*color_vbo)
//{
//    OpenGLVertexBuffer* openglvbo0 = static_cast<OpenGLVertexBuffer*>(vbo0);
//    OpenGLVertexBuffer* openglvbo1 = static_cast<OpenGLVertexBuffer*>(vbo1);
//    OpenGLVertexBuffer* openglvbo2 = static_cast<OpenGLVertexBuffer*>(vbo2);
//    OpenGLVertexBuffer* openglmodel = static_cast<OpenGLVertexBuffer*>(model_vbo);
//    OpenGLVertexBuffer* openglcolor = static_cast<OpenGLVertexBuffer*>(color_vbo);
//
//    CHECK_ERROR(glGenVertexArrays(1, vao));
//    CHECK_ERROR(glBindVertexArray(*vao));
//    CHECK_ERROR(glGenBuffers(1, &openglvbo0->mBuffer));      // vertex vbo
//    CHECK_ERROR(glGenBuffers(1, &openglvbo1->mBuffer));      // normals vbo
//    CHECK_ERROR(glGenBuffers(1, &openglvbo2->mBuffer));      // texcoords vbo
//    CHECK_ERROR(glGenBuffers(1, &openglmodel->mBuffer)); // instance model vbo
//    CHECK_ERROR(glGenBuffers(1, &openglcolor->mBuffer)); // instance color vbo
//
//    CHECK_ERROR(glBindVertexArray(*vao));
//    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, openglvbo0->mBuffer));
//    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW));
//    CHECK_ERROR(glEnableVertexAttribArray(0));
//    CHECK_ERROR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0));
//
//    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, openglvbo1->mBuffer));
//    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_DYNAMIC_DRAW));
//    CHECK_ERROR(glEnableVertexAttribArray(1));
//    CHECK_ERROR(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0));
//
//    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, openglvbo2->mBuffer));
//    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, texCoords.size() * sizeof(float), texCoords.data(), GL_DYNAMIC_DRAW));
//    CHECK_ERROR(glEnableVertexAttribArray(2));
//    CHECK_ERROR(glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0));
//
//    // instancing model matrices vbo
//    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, openglmodel->mBuffer));
//    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, INSTANCE_BATCH_SIZE * sizeof(glm::mat4), NULL, GL_DYNAMIC_DRAW));
//    // set attribute pointers for matrix (4 times vec4)
//    CHECK_ERROR(glEnableVertexAttribArray(3));
//    CHECK_ERROR(glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)0));
//    CHECK_ERROR(glEnableVertexAttribArray(4));
//    CHECK_ERROR(glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(sizeof(glm::vec4))));
//    CHECK_ERROR(glEnableVertexAttribArray(5));
//    CHECK_ERROR(glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(2 * sizeof(glm::vec4))));
//    CHECK_ERROR(glEnableVertexAttribArray(6));
//    CHECK_ERROR(glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(3 * sizeof(glm::vec4))));
//
//    CHECK_ERROR(glVertexAttribDivisor(3, 1));
//    CHECK_ERROR(glVertexAttribDivisor(4, 1));
//    CHECK_ERROR(glVertexAttribDivisor(5, 1));
//    CHECK_ERROR(glVertexAttribDivisor(6, 1));
//
//    // instancing colors vbo
//    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, openglcolor->mBuffer));
//    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, INSTANCE_BATCH_SIZE * sizeof(glm::vec4), NULL, GL_DYNAMIC_DRAW));
//    CHECK_ERROR(glEnableVertexAttribArray(7));
//    CHECK_ERROR(glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void *)0));
//
//    CHECK_ERROR(glVertexAttribDivisor(7, 1));
//
//    CHECK_ERROR(glBindVertexArray(0));
//}
//
//void OpenGLRenderer::destroyMesh_impl(unsigned int *vao, VertexBuffer*vbo0, VertexBuffer*vbo1, VertexBuffer*vbo2,
//    VertexBuffer*model_vbo, VertexBuffer*color_vbo)
//{
//    OpenGLVertexBuffer* openglvbo0 = static_cast<OpenGLVertexBuffer*>(vbo0);
//    OpenGLVertexBuffer* openglvbo1 = static_cast<OpenGLVertexBuffer*>(vbo1);
//    OpenGLVertexBuffer* openglvbo2 = static_cast<OpenGLVertexBuffer*>(vbo2);
//    OpenGLVertexBuffer* openglmodel = static_cast<OpenGLVertexBuffer*>(model_vbo);
//    OpenGLVertexBuffer* openglcolor = static_cast<OpenGLVertexBuffer*>(color_vbo);
//
//    CHECK_ERROR(glDeleteBuffers(1, &openglvbo0->mBuffer));
//    CHECK_ERROR(glDeleteBuffers(1, &openglvbo1->mBuffer));
//    CHECK_ERROR(glDeleteBuffers(1, &openglvbo2->mBuffer));
//    CHECK_ERROR(glDeleteBuffers(1, &openglmodel->mBuffer));
//    CHECK_ERROR(glDeleteBuffers(1, &openglcolor->mBuffer));
//
//    CHECK_ERROR(glDeleteVertexArrays(1, vao));
//}

void OpenGLRenderer::updateInstanceBuffer_impl(unsigned int vbo, const glm::mat4 *models, size_t instanceCount)
{
    assert(instanceCount <= INSTANCE_BATCH_SIZE);

    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo));
    CHECK_ERROR(glBufferSubData(GL_ARRAY_BUFFER, 0, instanceCount * sizeof(glm::mat4), models));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void OpenGLRenderer::updateInstanceColorBuffer_impl(unsigned int vbo, const glm::vec4 *colors, size_t instanceCount)
{
    assert(instanceCount <= INSTANCE_BATCH_SIZE);

    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo));
    CHECK_ERROR(glBufferSubData(GL_ARRAY_BUFFER, 0, instanceCount * sizeof(glm::vec4), colors));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void OpenGLRenderer::createSprite_impl(unsigned int *vao)
{
    // configure VAO/VBO
    unsigned int vbo;
    float vertices[] = {// pos      // tex
                        0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

                        0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f};

    CHECK_ERROR(glGenVertexArrays(1, vao));
    CHECK_ERROR(glGenBuffers(1, &vbo));

    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo));
    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW));

    CHECK_ERROR(glBindVertexArray(*vao));
    CHECK_ERROR(glEnableVertexAttribArray(0));
    CHECK_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLRenderer::destroySprite_impl(unsigned int *vao)
{
    CHECK_ERROR(glDeleteVertexArrays(1, vao));
}

void OpenGLRenderer::createFrustum_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
                                           unsigned int *vao,
                             unsigned int *vbo0, unsigned int *vbo1)
{
    CHECK_ERROR(glGenVertexArrays(1, vao));
    CHECK_ERROR(glBindVertexArray(*vao));

    CHECK_ERROR(glGenBuffers(2, vbo0));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, *vbo0));
    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_DYNAMIC_DRAW));
    CHECK_ERROR(glEnableVertexAttribArray(0));
    CHECK_ERROR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0));

    CHECK_ERROR(glGenBuffers(1, vbo1));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, *vbo1));
    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), &normals[0], GL_DYNAMIC_DRAW));
    CHECK_ERROR(glEnableVertexAttribArray(1));
    CHECK_ERROR(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0));

    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLRenderer::destroyFrustum_impl(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1)
{
    CHECK_ERROR(glDeleteVertexArrays(1, vao));
    CHECK_ERROR(glDeleteBuffers(2, vbo0));
    CHECK_ERROR(glDeleteBuffers(2, vbo1));
}

void OpenGLRenderer::updateFrustum_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
                                           unsigned int vbo0,
                             unsigned int vbo1)
{
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo0));
    CHECK_ERROR(glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), &vertices[0]));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));

    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo1));
    CHECK_ERROR(glBufferSubData(GL_ARRAY_BUFFER, 0, normals.size() * sizeof(float), &normals[0]));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void OpenGLRenderer::updateFrustum_impl(const std::vector<float> &vertices, unsigned int vbo0)
{
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, vbo0));
    CHECK_ERROR(glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), &vertices[0]));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void OpenGLRenderer::createGrid_impl(const std::vector<glm::vec3> &vertices, unsigned int *vao, unsigned int *vbo0)
{
    CHECK_ERROR(glGenVertexArrays(1, vao));
    CHECK_ERROR(glBindVertexArray(*vao));

    CHECK_ERROR(glGenBuffers(1, vbo0));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, *vbo0));
    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW));
    CHECK_ERROR(glEnableVertexAttribArray(0));
    CHECK_ERROR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0));

    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLRenderer::destroyGrid_impl(unsigned int *vao, unsigned int *vbo0)
{
    CHECK_ERROR(glDeleteVertexArrays(1, vao));
    CHECK_ERROR(glDeleteBuffers(1, vbo0));
}

void OpenGLRenderer::createLine_impl(const std::vector<float> &vertices, const std::vector<float> &colors,
                                        unsigned int *vao,
                          unsigned int *vbo0, unsigned int *vbo1)
{
    CHECK_ERROR(glGenVertexArrays(1, vao));
    CHECK_ERROR(glBindVertexArray(*vao));

    CHECK_ERROR(glGenBuffers(2, vbo0));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, *vbo0));
    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW));
    CHECK_ERROR(glEnableVertexAttribArray(0));
    CHECK_ERROR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0));

    CHECK_ERROR(glGenBuffers(1, vbo1));
    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, *vbo1));
    CHECK_ERROR(glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float), &colors[0], GL_STATIC_DRAW));
    CHECK_ERROR(glEnableVertexAttribArray(1));
    CHECK_ERROR(glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GL_FLOAT), 0));

    CHECK_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLRenderer::destroyLine_impl(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1)
{
    CHECK_ERROR(glDeleteVertexArrays(1, vao));
    CHECK_ERROR(glDeleteBuffers(1, vbo0));
    CHECK_ERROR(glDeleteBuffers(1, vbo1));
}

void OpenGLRenderer::preprocess_impl(std::string &vert, std::string &frag, std::string &geom, int64_t variant)
{
    std::string version;
    std::string defines;
    std::string shader;

    if (variant & static_cast<int64_t>(ShaderMacro::Directional))
    {
        defines += "#define DIRECTIONALLIGHT\n";
    }
    if (variant & static_cast<int64_t>(ShaderMacro::Spot))
    {
        defines += "#define SPOTLIGHT\n";
    }
    if (variant & static_cast<int64_t>(ShaderMacro::Point))
    {
        defines += "#define POINTLIGHT\n";
    }
    if (variant & static_cast<int64_t>(ShaderMacro::HardShadows))
    {
        defines += "#define HARDSHADOWS\n";
    }
    if (variant & static_cast<int64_t>(ShaderMacro::SoftShadows))
    {
        defines += "#define SOFTSHADOWS\n";
    }
    if (variant & static_cast<int64_t>(ShaderMacro::SSAO))
    {
        defines += "#define SSAO\n";
    }
    if (variant & static_cast<int64_t>(ShaderMacro::ShowCascades))
    {
        defines += "#define SHOWCASCADES\n";
    }
    if (variant & static_cast<int64_t>(ShaderMacro::Instancing))
    {
        defines += "#define INSTANCING\n";
    }

    size_t pos = vert.find('\n');
    if (pos != std::string::npos)
    {
        version = vert.substr(0, pos + 1);
        shader = vert.substr(pos + 1);
    }

    vert = version + defines + shader;

    pos = frag.find('\n');
    if (pos != std::string::npos)
    {
        version = frag.substr(0, pos + 1);
        shader = frag.substr(pos + 1);
    }

    frag = version + defines + shader;

    // pos = geom.find('\n');
    // if (pos != std::string::npos)
    //{
    //     version = geom.substr(0, pos + 1);
    //     shader = geom.substr(pos + 1);
    // }

    // geom = version + defines + shader;
}

void OpenGLRenderer::compile_impl(const std::string &name, const std::string &vert, const std::string &frag,
                       const std::string &geom, unsigned int *program, ShaderStatus &status)
{
    memset(status.mVertexCompileLog, 0, sizeof(status.mVertexCompileLog));
    memset(status.mFragmentCompileLog, 0, sizeof(status.mFragmentCompileLog));
    memset(status.mGeometryCompileLog, 0, sizeof(status.mGeometryCompileLog));
    memset(status.mLinkLog, 0, sizeof(status.mLinkLog));

    const GLchar *vertexShaderCharPtr = vert.c_str();
    const GLchar *geometryShaderCharPtr = geom.c_str();
    const GLchar *fragmentShaderCharPtr = frag.c_str();

    // Compile vertex shader
    GLuint vertexShaderObj = glCreateShader(GL_VERTEX_SHADER);
    CHECK_ERROR(glShaderSource(vertexShaderObj, 1, &vertexShaderCharPtr, NULL));
    CHECK_ERROR(glCompileShader(vertexShaderObj));
    CHECK_ERROR(glGetShaderiv(vertexShaderObj, GL_COMPILE_STATUS, &status.mVertexShaderCompiled));
    if (!status.mVertexShaderCompiled)
    {
        CHECK_ERROR(glGetShaderInfoLog(vertexShaderObj, 512, NULL, status.mVertexCompileLog));

        std::string message = "Shader: Vertex shader compilation failed (" + name + ")\n";
        Log::error(message.c_str());
    }

    // Compile fragment shader
    GLuint fragmentShaderObj = glCreateShader(GL_FRAGMENT_SHADER);
    CHECK_ERROR(glShaderSource(fragmentShaderObj, 1, &fragmentShaderCharPtr, NULL));
    CHECK_ERROR(glCompileShader(fragmentShaderObj));
    CHECK_ERROR(glGetShaderiv(fragmentShaderObj, GL_COMPILE_STATUS, &status.mFragmentShaderCompiled));
    if (!status.mFragmentShaderCompiled)
    {
        CHECK_ERROR(glGetShaderInfoLog(fragmentShaderObj, 512, NULL, status.mFragmentCompileLog));

        std::string message = "Shader: Fragment shader compilation failed (" + name + ")\n";
        Log::error(message.c_str());
    }

    // Compile geometry shader
    GLuint geometryShaderObj = 0;
    if (!geom.empty())
    {
        geometryShaderObj = glCreateShader(GL_GEOMETRY_SHADER);
        CHECK_ERROR(glShaderSource(geometryShaderObj, 1, &geometryShaderCharPtr, NULL));
        CHECK_ERROR(glCompileShader(geometryShaderObj));
        CHECK_ERROR(glGetShaderiv(geometryShaderObj, GL_COMPILE_STATUS, &status.mGeometryShaderCompiled));
        if (!status.mGeometryShaderCompiled)
        {
            CHECK_ERROR(glGetShaderInfoLog(geometryShaderObj, 512, NULL, status.mGeometryCompileLog));

            std::string message = "Shader: Geometry shader compilation failed (" + name + ")\n";
            Log::error(message.c_str());
        }
    }

    // Create shader program
    *program = glCreateProgram();

    // Attach shader objects to shader program
    CHECK_ERROR(glAttachShader(*program, vertexShaderObj));
    CHECK_ERROR(glAttachShader(*program, fragmentShaderObj));
    if (geometryShaderObj != 0)
    {
        CHECK_ERROR(glAttachShader(*program, geometryShaderObj));
    }

    // Link shader program
    CHECK_ERROR(glLinkProgram(*program));
    CHECK_ERROR(glGetProgramiv(*program, GL_LINK_STATUS, &status.mShaderLinked));
    if (!status.mShaderLinked)
    {
        CHECK_ERROR(glGetProgramInfoLog(*program, 512, NULL, status.mLinkLog));

        std::string message = "Shader: " + name + " program linking failed\n";
        Log::error(message.c_str());
    }

    // Detach shader objects from shader program
    CHECK_ERROR(glDetachShader(*program, vertexShaderObj));
    CHECK_ERROR(glDetachShader(*program, fragmentShaderObj));
    if (geometryShaderObj != 0)
    {
        CHECK_ERROR(glDetachShader(*program, geometryShaderObj));
    }

    // Delete shader objects
    CHECK_ERROR(glDeleteShader(vertexShaderObj));
    CHECK_ERROR(glDeleteShader(fragmentShaderObj));
    if (!geom.empty())
    {
        CHECK_ERROR(glDeleteShader(geometryShaderObj));
    }
}

int OpenGLRenderer::findUniformLocation_impl(const char *name, int program)
{
    return glGetUniformLocation(program, name);
}

int OpenGLRenderer::getUniformCount_impl(int program)
{
    GLint uniformCount;
    CHECK_ERROR(glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &uniformCount));

    return uniformCount;
}

int OpenGLRenderer::getAttributeCount_impl(int program)
{
    GLint attributeCount;
    CHECK_ERROR(glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &attributeCount));

    return attributeCount;
}

std::vector<ShaderUniform> OpenGLRenderer::getShaderUniforms_impl(int program)
{
    GLint uniformCount;
    CHECK_ERROR(glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &uniformCount));

    std::vector<ShaderUniform> uniforms(uniformCount);

    for (int j = 0; j < uniformCount; j++)
    {
        Uniform uniform;
        CHECK_ERROR(glGetActiveUniform(program, (GLuint)j, 32, &uniform.nameLength, &uniform.size, &uniform.type, &uniform.name[0]));

        uniforms[j].mName = std::string(uniform.name);
        switch (uniform.type)
        {
        case GL_INT:
            uniforms[j].mType = ShaderUniformType::Int;
            break;
        case GL_FLOAT:
            uniforms[j].mType = ShaderUniformType::Float;
            break;
        case GL_FLOAT_VEC2:
            uniforms[j].mType = ShaderUniformType::Vec2;
            break;
        case GL_FLOAT_VEC3:
            uniforms[j].mType = ShaderUniformType::Vec3;
            break;
        case GL_FLOAT_VEC4:
            uniforms[j].mType = ShaderUniformType::Vec4;
            break;
        case GL_FLOAT_MAT2:
            uniforms[j].mType = ShaderUniformType::Mat2;
            break;
        case GL_FLOAT_MAT3:
            uniforms[j].mType = ShaderUniformType::Mat3;
            break;
        case GL_FLOAT_MAT4:
            uniforms[j].mType = ShaderUniformType::Mat4;
            break;
        case GL_SAMPLER_2D:
            uniforms[j].mType = ShaderUniformType::Sampler2D;
            break;
        case GL_SAMPLER_CUBE:
            uniforms[j].mType = ShaderUniformType::SamplerCube;
            break;
        }

        uniforms[j].mUniformId = 0;
        //uniforms[j].mTex = -1;
        uniforms[j].mTex = nullptr;
        memset(uniforms[j].mData, '\0', 64);
    }

    return uniforms;
}

std::vector<ShaderAttribute> OpenGLRenderer::getShaderAttributes_impl(int program)
{
    GLint attributeCount;
    CHECK_ERROR(glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &attributeCount));

    std::vector<ShaderAttribute> attributes(attributeCount);

    for (int j = 0; j < attributeCount; j++)
    {
        Attribute attrib;
        CHECK_ERROR(glGetActiveAttrib(program, (GLuint)j, 32, &attrib.nameLength, &attrib.size, &attrib.type, &attrib.name[0]));

        attributes[j].mName = std::string(attrib.name);
    }

    return attributes;
}

void OpenGLRenderer::setUniformBlock_impl(const char *blockName, int bindingPoint, int program)
{
    GLuint blockIndex = glGetUniformBlockIndex(program, blockName);
    if (blockIndex != GL_INVALID_INDEX)
    {
        CHECK_ERROR(glUniformBlockBinding(program, blockIndex, bindingPoint));
    }
}

//void OpenGLRenderer::use_impl(int program)
//{
//    CHECK_ERROR(glUseProgram(program));
//}
//
//void OpenGLRenderer::unuse_impl()
//{
//    CHECK_ERROR(glUseProgram(0));
//}
//
//void OpenGLRenderer::destroy_impl(int program)
//{
//    CHECK_ERROR(glDeleteProgram(program));
//}
//
//void OpenGLRenderer::setBool_impl(int nameLocation, bool value)
//{
//    CHECK_ERROR(glUniform1i(nameLocation, (int)value));
//}
//
//void OpenGLRenderer::setInt_impl(int nameLocation, int value)
//{
//    CHECK_ERROR(glUniform1i(nameLocation, value));
//}
//
//void OpenGLRenderer::setFloat_impl(int nameLocation, float value)
//{
//    CHECK_ERROR(glUniform1f(nameLocation, value));
//}
//
//void OpenGLRenderer::setColor_impl(int nameLocation, const Color &color)
//{
//    CHECK_ERROR(glUniform4fv(nameLocation, 1, static_cast<const GLfloat *>(&color.mR)));
//}
//
//void OpenGLRenderer::setColor32_impl(int nameLocation, const Color32 &color)
//{
//    CHECK_ERROR(glUniform4ui(nameLocation, static_cast<GLuint>(color.mR), static_cast<GLuint>(color.mG),
//                 static_cast<GLuint>(color.mB), static_cast<GLuint>(color.mA)));
//}
//
//void OpenGLRenderer::setVec2_impl(int nameLocation, const glm::vec2 &vec)
//{
//    CHECK_ERROR(glUniform2fv(nameLocation, 1, &vec[0]));
//}
//
//void OpenGLRenderer::setVec3_impl(int nameLocation, const glm::vec3 &vec)
//{
//    CHECK_ERROR(glUniform3fv(nameLocation, 1, &vec[0]));
//}
//
//void OpenGLRenderer::setVec4_impl(int nameLocation, const glm::vec4 &vec)
//{
//    CHECK_ERROR(glUniform4fv(nameLocation, 1, &vec[0]));
//}
//
//void OpenGLRenderer::setMat2_impl(int nameLocation, const glm::mat2 &mat)
//{
//    CHECK_ERROR(glUniformMatrix2fv(nameLocation, 1, GL_FALSE, &mat[0][0]));
//}
//
//void OpenGLRenderer::setMat3_impl(int nameLocation, const glm::mat3 &mat)
//{
//    CHECK_ERROR(glUniformMatrix3fv(nameLocation, 1, GL_FALSE, &mat[0][0]));
//}
//
//void OpenGLRenderer::setMat4_impl(int nameLocation, const glm::mat4 &mat)
//{
//    CHECK_ERROR(glUniformMatrix4fv(nameLocation, 1, GL_FALSE, &mat[0][0]));
//}
//
//void OpenGLRenderer::setTexture2D_impl(int nameLocation, int texUnit, TextureHandle* tex)
//{
//    CHECK_ERROR(glUniform1i(nameLocation, texUnit));
//
//    CHECK_ERROR(glActiveTexture(GL_TEXTURE0 + texUnit));
//    if (tex != nullptr)
//    {
//        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *reinterpret_cast<unsigned int*>(tex->getHandle())));
//    }
//    else
//    {
//        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
//    }
//}
//
//void OpenGLRenderer::setTexture2Ds_impl(int nameLocation, const std::vector<int>& texUnits, int count, const std::vector<TextureHandle*>& texs)
//{
//    CHECK_ERROR(glUniform1iv(nameLocation, count, texUnits.data()));
//
//    for (int i = 0; i < count; i++)
//    {
//        CHECK_ERROR(glActiveTexture(GL_TEXTURE0 + texUnits[i]));
//        CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, *reinterpret_cast<unsigned int*>(texs[i]->getHandle())));
//    }
//}
//
//bool OpenGLRenderer::getBool_impl(int nameLocation, int program)
//{
//    int value = 0;
//    CHECK_ERROR(glGetUniformiv(program, nameLocation, &value));
//
//    return (bool)value;
//}
//
//int OpenGLRenderer::getInt_impl(int nameLocation, int program)
//{
//    int value = 0;
//    CHECK_ERROR(glGetUniformiv(program, nameLocation, &value));
//
//    return value;
//}
//
//float OpenGLRenderer::getFloat_impl(int nameLocation, int program)
//{
//    float value = 0.0f;
//    CHECK_ERROR(glGetUniformfv(program, nameLocation, &value));
//
//    return value;
//}
//
//Color OpenGLRenderer::getColor_impl(int nameLocation, int program)
//{
//    Color color = Color(0.0f, 0.0f, 0.0f, 1.0f);
//    CHECK_ERROR(glGetnUniformfv(program, nameLocation, sizeof(Color), &color.mR));
//
//    return color;
//}
//
//Color32 OpenGLRenderer::getColor32_impl(int nameLocation, int program)
//{
//    Color32 color = Color32(0, 0, 0, 255);
//
//    GLuint c[4];
//    CHECK_ERROR(glGetnUniformuiv(program, nameLocation, 4 * sizeof(GLuint), &c[0]));
//
//    color.mR = static_cast<unsigned char>(c[0]);
//    color.mG = static_cast<unsigned char>(c[1]);
//    color.mB = static_cast<unsigned char>(c[2]);
//    color.mA = static_cast<unsigned char>(c[3]);
//
//    return color;
//}
//
//glm::vec2 OpenGLRenderer::getVec2_impl(int nameLocation, int program)
//{
//    glm::vec2 value = glm::vec2(0.0f);
//    CHECK_ERROR(glGetnUniformfv(program, nameLocation, sizeof(glm::vec2), &value[0]));
//
//    return value;
//}
//
//glm::vec3 OpenGLRenderer::getVec3_impl(int nameLocation, int program)
//{
//    glm::vec3 value = glm::vec3(0.0f);
//    CHECK_ERROR(glGetnUniformfv(program, nameLocation, sizeof(glm::vec3), &value[0]));
//
//    return value;
//}
//
//glm::vec4 OpenGLRenderer::getVec4_impl(int nameLocation, int program)
//{
//    glm::vec4 value = glm::vec4(0.0f);
//    CHECK_ERROR(glGetnUniformfv(program, nameLocation, sizeof(glm::vec4), &value[0]));
//
//    return value;
//}
//
//glm::mat2 OpenGLRenderer::getMat2_impl(int nameLocation, int program)
//{
//    glm::mat2 value = glm::mat2(0.0f);
//    CHECK_ERROR(glGetnUniformfv(program, nameLocation, sizeof(glm::mat2), &value[0][0]));
//
//    return value;
//}
//
//glm::mat3 OpenGLRenderer::getMat3_impl(int nameLocation, int program)
//{
//    glm::mat3 value = glm::mat3(0.0f);
//    CHECK_ERROR(glGetnUniformfv(program, nameLocation, sizeof(glm::mat3), &value[0][0]));
//
//    return value;
//}
//
//glm::mat4 OpenGLRenderer::getMat4_impl(int nameLocation, int program)
//{
//    glm::mat4 value = glm::mat4(0.0f);
//    CHECK_ERROR(glGetnUniformfv(program, nameLocation, sizeof(glm::mat4), &value[0][0]));
//
//    return value;
//}

void OpenGLRenderer::applyMaterial_impl(const std::vector<ShaderUniform> &uniforms, ShaderProgram* shaderProgram)
{
    
    int textureUnit = 0;
    for (size_t i = 0; i < uniforms.size(); i++)
    {
        int location = findUniformLocation(uniforms[i].mName.c_str(), *reinterpret_cast<unsigned int*>(shaderProgram->getHandle()));

        assert(location != -1);

        if (uniforms[i].mType == ShaderUniformType::Sampler2D)
        {
            if (uniforms[i].mTex != nullptr)
            {
                shaderProgram->setTexture2D(location, textureUnit, uniforms[i].mTex);
            }
            else
            {
                shaderProgram->setTexture2D(location, textureUnit, nullptr);
            }

            textureUnit++;
        }
        else if (uniforms[i].mType == ShaderUniformType::Int)
        {
            shaderProgram->setInt(location, *reinterpret_cast<const int*>(uniforms[i].mData));
        }
        else if (uniforms[i].mType == ShaderUniformType::Float)
        {
            shaderProgram->setFloat(location, *reinterpret_cast<const float*>(uniforms[i].mData));
        }
        else if (uniforms[i].mType == ShaderUniformType::Vec2)
        {
            shaderProgram->setVec2(location, *reinterpret_cast<const glm::vec2*>(uniforms[i].mData));
        }
        else if (uniforms[i].mType == ShaderUniformType::Vec3)
        {
            shaderProgram->setVec3(location, *reinterpret_cast<const glm::vec3*>(uniforms[i].mData));
        }
        else if (uniforms[i].mType == ShaderUniformType::Vec4)
        {
            shaderProgram->setVec4(location, *reinterpret_cast<const glm::vec4*>(uniforms[i].mData));
        }
    }
}

void OpenGLRenderer::renderLines_impl(int start, int count, int vao)
{
    CHECK_ERROR(glBindVertexArray(vao));
    CHECK_ERROR(glDrawArrays(GL_LINES, start, count));
    CHECK_ERROR(glBindVertexArray(0));
}

void OpenGLRenderer::renderLinesWithCurrentlyBoundVAO_impl(int start, int count)
{
    CHECK_ERROR(glDrawArrays(GL_LINES, start, count));
}

void OpenGLRenderer::renderWithCurrentlyBoundVAO_impl(int start, int count)
{
    CHECK_ERROR(glDrawArrays(GL_TRIANGLES, start, count));
}

void OpenGLRenderer::render_impl(int start, int count, int vao, bool wireframe)
{
    if (wireframe)
    {
        CHECK_ERROR(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
    }

    CHECK_ERROR(glBindVertexArray(vao));
    CHECK_ERROR(glDrawArrays(GL_TRIANGLES, start, count));
    CHECK_ERROR(glBindVertexArray(0));

    if (wireframe)
    {
        CHECK_ERROR(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
    }
}

void OpenGLRenderer::render_impl(int start, int count, int vao, GraphicsQuery &query, bool wireframe)
{
    if (wireframe)
    {
        CHECK_ERROR(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
    }

    CHECK_ERROR(glBindVertexArray(vao));
    CHECK_ERROR(glDrawArrays(GL_TRIANGLES, start, count));
    CHECK_ERROR(glBindVertexArray(0));

    if (wireframe)
    {
        CHECK_ERROR(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
    }

    query.mNumDrawCalls++;
    query.mVerts += count;
    query.mTris += count / 3;
}

void OpenGLRenderer::renderInstanced_impl(int start, int count, int instanceCount, int vao, GraphicsQuery &query)
{
    CHECK_ERROR(glBindVertexArray(vao));
    CHECK_ERROR(glDrawArraysInstanced(GL_TRIANGLES, start, count, instanceCount));
    CHECK_ERROR(glBindVertexArray(0));

    query.mNumBatchDrawCalls++;
    query.mVerts += count;
    query.mTris += count / 3;    
}

void OpenGLRenderer::render_impl(const RenderObject &renderObject, GraphicsQuery &query)
{
    assert(renderObject.instanced == false);

    OpenGLRenderer::render(renderObject.start / 3, renderObject.size / 3, renderObject.vao, query);
}

void OpenGLRenderer::renderInstanced_impl(const RenderObject &renderObject, GraphicsQuery &query)
{
    assert(renderObject.instanced == true);

    OpenGLRenderer::renderInstanced(renderObject.start / 3, renderObject.size / 3, renderObject.instanceCount,
                              renderObject.vao, query);
}
