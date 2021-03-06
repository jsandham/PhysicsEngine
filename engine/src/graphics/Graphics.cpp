#include <GL/glew.h>
#include <algorithm>
#include <iostream>
#include <random>

#include "../../include/core/InternalShaders.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

void Graphics::checkError(long line, const char *file)
{
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR)
    {
        std::string errorStr;
        switch (error)
        {
        case GL_INVALID_ENUM:
            errorStr = "An unacceptable value is specified for an enumerated argument";
            break;
        case GL_INVALID_VALUE:
            errorStr = "A numeric argument is out of range";
            break;
        case GL_INVALID_OPERATION:
            errorStr = "The specified operation is not allowed in the current state";
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            errorStr = "The framebuffer object is not complete";
            break;
        case GL_OUT_OF_MEMORY:
            errorStr = "There is not enough money left to execute the command";
            break;
        case GL_STACK_UNDERFLOW:
            errorStr = "An attempt has been made to perform an operation that would cause an internal stack to "
                       "underflow";
            break;
        case GL_STACK_OVERFLOW:
            errorStr = "An attempt has been made to perform an operation that would cause an internal stack to "
                       "overflow";
            break;
        default:
            errorStr = "Unknown error";
            break;
        }

        std::string errorMessage =
            errorStr + "(" + std::to_string(error) + ") line: " + std::to_string(line) + " file: " + file + "\n";
        Log::error(errorMessage.c_str());
    }
}

void Graphics::checkFrambufferError(long line, const char *file)
{
    GLenum framebufferStatus = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
    if (framebufferStatus != GL_FRAMEBUFFER_COMPLETE)
    {
        std::string errorStr;
        switch (framebufferStatus)
        {
        case GL_FRAMEBUFFER_UNDEFINED:
            errorStr = "The current FBO binding is 0 but no default framebuffer exists";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            errorStr = "One of the buffers enabled for rendering is incomplete";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            errorStr = "No buffers are attached to the FBO and it is not configured for rendering without attachments";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
            errorStr = "Not all attachments enabled via glDrawBuffers exists in framebuffer";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
            errorStr = "Not all buffers specified via glReadBuffer exists in framebuffer";
            break;
        case GL_FRAMEBUFFER_UNSUPPORTED:
            errorStr = "The combination of internal buffer formats is unsupported";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
            errorStr = "The number of samples for each attachment is not the same";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
            errorStr = "Not all color attachments are layered textures or bound to the same target";
            break;
        default:
            errorStr = "Unknown framebuffer status error";
            break;
        }

        std::string errorMessage = errorStr + "(" + std::to_string(framebufferStatus) +
                                   ") line: " + std::to_string(line) + " file: " + file + "\n";
        Log::error(errorMessage.c_str());
    }
}

GLenum Graphics::getTextureFormat(TextureFormat format)
{
    GLenum openglFormat = GL_DEPTH_COMPONENT;

    switch (format)
    {
    case Depth:
        openglFormat = GL_DEPTH_COMPONENT;
        break;
    case RG:
        openglFormat = GL_RG;
        break;
    case RGB:
        openglFormat = GL_RGB;
        break;
    case RGBA:
        openglFormat = GL_RGBA;
        break;
    default:
        Log::error("OpengGL: Invalid texture format\n");
        break;
    }

    return openglFormat;
}

GLint Graphics::getTextureWrapMode(TextureWrapMode wrapMode)
{
    GLint openglWrapMode = GL_REPEAT;

    switch (wrapMode)
    {
    case Repeat:
        openglWrapMode = GL_REPEAT;
        break;
    case Clamp:
        openglWrapMode = GL_CLAMP_TO_EDGE;
        break;
    default:
        Log::error("OpengGL: Invalid texture wrap mode\n");
        break;
    }

    return openglWrapMode;
}

GLint Graphics::getTextureFilterMode(TextureFilterMode filterMode)
{
    GLint openglFilterMode = GL_NEAREST;

    switch (filterMode)
    {
    case Nearest:
        openglFilterMode = GL_NEAREST;
        break;
    case Bilinear:
        openglFilterMode = GL_LINEAR;
        break;
    case Trilinear:
        openglFilterMode = GL_LINEAR_MIPMAP_LINEAR;
        break;
    default:
        Log::error("OpengGL: Invalid texture filter mode\n");
        break;
    }

    return openglFilterMode;
}

void Graphics::beginQuery(GLuint queryId)
{
    glBeginQuery(GL_TIME_ELAPSED, queryId);
}

void Graphics::endQuery(GLuint queryId, GLuint64 *elapsedTime)
{
    glEndQuery(GL_TIME_ELAPSED);
    glGetQueryObjectui64v(queryId, GL_QUERY_RESULT, elapsedTime);
}

void Graphics::createGlobalCameraUniforms(CameraUniform &uniform)
{
    glGenBuffers(1, &uniform.mBuffer);
    glBindBuffer(GL_UNIFORM_BUFFER, uniform.mBuffer);
    glBufferData(GL_UNIFORM_BUFFER, 144, NULL, GL_DYNAMIC_DRAW);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, uniform.mBuffer, 0, 144);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createGlobalLightUniforms(LightUniform &uniform)
{
    glGenBuffers(1, &uniform.mBuffer);
    glBindBuffer(GL_UNIFORM_BUFFER, uniform.mBuffer);
    glBufferData(GL_UNIFORM_BUFFER, 824, NULL, GL_DYNAMIC_DRAW);
    glBindBufferRange(GL_UNIFORM_BUFFER, 1, uniform.mBuffer, 0, 824);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::setGlobalCameraUniforms(const CameraUniform &uniform)
{
    glBindBuffer(GL_UNIFORM_BUFFER, uniform.mBuffer);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, uniform.mBuffer, 0, 144);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, 64, glm::value_ptr(uniform.mProjection));
    glBufferSubData(GL_UNIFORM_BUFFER, 64, 64, glm::value_ptr(uniform.mView));
    glBufferSubData(GL_UNIFORM_BUFFER, 128, 12, glm::value_ptr(uniform.mCameraPos));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::setGlobalLightUniforms(const LightUniform &uniform)
{
    glBindBuffer(GL_UNIFORM_BUFFER, uniform.mBuffer);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, 320, &uniform.mLightProjection[0]);
    glBufferSubData(GL_UNIFORM_BUFFER, 320, 320, &uniform.mLightView[0]);
    glBufferSubData(GL_UNIFORM_BUFFER, 640, 12, glm::value_ptr(uniform.mPosition));
    glBufferSubData(GL_UNIFORM_BUFFER, 656, 12, glm::value_ptr(uniform.mDirection));
    glBufferSubData(GL_UNIFORM_BUFFER, 672, 16, glm::value_ptr(uniform.mColor));
    glBufferSubData(GL_UNIFORM_BUFFER, 688, 4, &uniform.mCascadeEnds[0]);
    glBufferSubData(GL_UNIFORM_BUFFER, 704, 4, &uniform.mCascadeEnds[1]);
    glBufferSubData(GL_UNIFORM_BUFFER, 720, 4, &uniform.mCascadeEnds[2]);
    glBufferSubData(GL_UNIFORM_BUFFER, 736, 4, &uniform.mCascadeEnds[3]);
    glBufferSubData(GL_UNIFORM_BUFFER, 752, 4, &uniform.mCascadeEnds[4]);
    glBufferSubData(GL_UNIFORM_BUFFER, 768, 4, &uniform.mIntensity);
    glBufferSubData(GL_UNIFORM_BUFFER, 772, 4, &(uniform.mSpotAngle));
    glBufferSubData(GL_UNIFORM_BUFFER, 776, 4, &(uniform.mInnerSpotAngle));
    glBufferSubData(GL_UNIFORM_BUFFER, 780, 4, &(uniform.mShadowNearPlane));
    glBufferSubData(GL_UNIFORM_BUFFER, 784, 4, &(uniform.mShadowFarPlane));
    glBufferSubData(GL_UNIFORM_BUFFER, 788, 4, &(uniform.mShadowAngle));
    glBufferSubData(GL_UNIFORM_BUFFER, 792, 4, &(uniform.mShadowRadius));
    glBufferSubData(GL_UNIFORM_BUFFER, 796, 4, &(uniform.mShadowStrength));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createScreenQuad(GLuint *vao, GLuint *vbo)
{
    // generate screen quad for final rendering
    constexpr float quadVertices[] = {
        // positions        // texture Coords
        -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        1.0f,  1.0f, 0.0f, 1.0f, 1.0f, 1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,
    };

    glGenVertexArrays(1, vao);
    glBindVertexArray(*vao);

    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::renderScreenQuad(GLuint vao)
{
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

void Graphics::createFramebuffer(int width, int height, GLuint *fbo, GLuint *color, GLuint *depth)
{
    // generate fbo (color + depth)
    glGenFramebuffers(1, fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, *fbo);

    glGenTextures(1, color);
    glBindTexture(GL_TEXTURE_2D, *color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenTextures(1, depth);
    glBindTexture(GL_TEXTURE_2D, *depth);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *color, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, *depth, 0);

    // - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
    unsigned int mainAttachments[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, mainAttachments);

    Graphics::checkFrambufferError(__LINE__, __FILE__);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Graphics::destroyFramebuffer(GLuint *fbo, GLuint *color, GLuint *depth)
{
    // detach textures from their framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, *fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // delete frambuffer
    glDeleteFramebuffers(1, fbo);

    // delete textures
    glDeleteTextures(1, color);
    glDeleteTextures(1, depth);
}

void Graphics::bindFramebuffer(GLuint fbo)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
}

void Graphics::unbindFramebuffer()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Graphics::clearFrambufferColor(const Color &color)
{
    glClearColor(color.r, color.g, color.b, color.a);
    glClear(GL_COLOR_BUFFER_BIT);
}

void Graphics::clearFrambufferColor(float r, float g, float b, float a)
{
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT);
}

void Graphics::clearFramebufferDepth(float depth)
{
    glClearDepth(depth);
    glClear(GL_DEPTH_BUFFER_BIT);
}

void Graphics::setViewport(int x, int y, int width, int height)
{
    glViewport(x, y, width, height);
}

void Graphics::createTargets(CameraTargets *targets, Viewport viewport, glm::vec3 *ssaoSamples, GLuint *queryId0,
                             GLuint *queryId1)
{
    // generate timing queries
    glGenQueries(1, queryId0);
    glGenQueries(1, queryId1);

    // generate main camera fbo (color + depth)
    glGenFramebuffers(1, &(targets->mMainFBO));
    glBindFramebuffer(GL_FRAMEBUFFER, targets->mMainFBO);

    glGenTextures(1, &(targets->mColorTex));
    glBindTexture(GL_TEXTURE_2D, targets->mColorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenTextures(1, &(targets->mDepthTex));
    glBindTexture(GL_TEXTURE_2D, targets->mDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1920, 1080, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, targets->mColorTex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, targets->mDepthTex, 0);

    // - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
    unsigned int mainAttachments[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, mainAttachments);

    Graphics::checkFrambufferError(__LINE__, __FILE__);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // generate color picking fbo (color + depth)
    glGenFramebuffers(1, &(targets->mColorPickingFBO));
    glBindFramebuffer(GL_FRAMEBUFFER, targets->mColorPickingFBO);

    glGenTextures(1, &(targets->mColorPickingTex));
    glBindTexture(GL_TEXTURE_2D, targets->mColorPickingTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenTextures(1, &(targets->mColorPickingDepthTex));
    glBindTexture(GL_TEXTURE_2D, targets->mColorPickingDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1920, 1080, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, targets->mColorPickingTex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, targets->mColorPickingDepthTex, 0);

    // - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
    unsigned int colorPickingAttachments[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, colorPickingAttachments);

    Graphics::checkFrambufferError(__LINE__, __FILE__);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // generate geometry fbo
    glGenFramebuffers(1, &(targets->mGeometryFBO));
    glBindFramebuffer(GL_FRAMEBUFFER, targets->mGeometryFBO);

    glGenTextures(1, &(targets->mPositionTex));
    glBindTexture(GL_TEXTURE_2D, targets->mPositionTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenTextures(1, &(targets->mNormalTex));
    glBindTexture(GL_TEXTURE_2D, targets->mNormalTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenTextures(1, &(targets->mAlbedoSpecTex));
    glBindTexture(GL_TEXTURE_2D, targets->mAlbedoSpecTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, targets->mPositionTex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, targets->mNormalTex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, targets->mAlbedoSpecTex, 0);

    unsigned int geometryAttachments[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, geometryAttachments);

    Graphics::checkFrambufferError(__LINE__, __FILE__);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // generate ssao fbo
    glGenFramebuffers(1, &(targets->mSsaoFBO));
    glBindFramebuffer(GL_FRAMEBUFFER, targets->mSsaoFBO);

    glGenTextures(1, &(targets->mSsaoColorTex));
    glBindTexture(GL_TEXTURE_2D, targets->mSsaoColorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, targets->mSsaoColorTex, 0);

    unsigned int ssaoAttachments[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, ssaoAttachments);

    Graphics::checkFrambufferError(__LINE__, __FILE__);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    auto lerp = [](float a, float b, float t) { return a + t * (b - a); };

    // generate noise texture for use in ssao
    std::uniform_real_distribution<GLfloat> distribution(0.0, 1.0);
    std::default_random_engine generator;
    for (unsigned int j = 0; j < 64; ++j)
    {
        float x = distribution(generator) * 2.0f - 1.0f;
        float y = distribution(generator) * 2.0f - 1.0f;
        float z = distribution(generator);
        float radius = distribution(generator);

        glm::vec3 sample(x, y, z);
        sample = radius * glm::normalize(sample);
        float scale = float(j) / 64.0f;

        // scale samples s.t. they're more aligned to center of kernel
        scale = lerp(0.1f, 1.0f, scale * scale);
        sample *= scale;

        ssaoSamples[j] = sample;
    }

    glm::vec3 ssaoNoise[16];
    for (int j = 0; j < 16; j++)
    {
        // rotate around z-axis (in tangent space)
        glm::vec3 noise(distribution(generator) * 2.0f - 1.0f, distribution(generator) * 2.0f - 1.0f, 0.0f);
        ssaoNoise[j] = noise;
    }

    glGenTextures(1, &(targets->mSsaoNoiseTex));
    glBindTexture(GL_TEXTURE_2D, targets->mSsaoNoiseTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssaoNoise[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindTexture(GL_TEXTURE_2D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::destroyTargets(CameraTargets *targets, GLuint *queryId0, GLuint *queryId1)
{
    // detach textures from their framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, targets->mMainFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, targets->mColorPickingFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, targets->mGeometryFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, targets->mSsaoFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // delete frambuffers
    glDeleteFramebuffers(1, &(targets->mMainFBO));
    glDeleteFramebuffers(1, &(targets->mColorPickingFBO));
    glDeleteFramebuffers(1, &(targets->mGeometryFBO));
    glDeleteFramebuffers(1, &(targets->mSsaoFBO));

    // delete textures
    glDeleteTextures(1, &(targets->mColorTex));
    glDeleteTextures(1, &(targets->mDepthTex));
    glDeleteTextures(1, &(targets->mColorPickingTex));
    glDeleteTextures(1, &(targets->mColorPickingDepthTex));
    glDeleteTextures(1, &(targets->mPositionTex));
    glDeleteTextures(1, &(targets->mNormalTex));
    glDeleteTextures(1, &(targets->mAlbedoSpecTex));
    glDeleteTextures(1, &(targets->mSsaoColorTex));
    glDeleteTextures(1, &(targets->mSsaoNoiseTex));

    // delete timing query
    glDeleteQueries(1, queryId0);
    glDeleteQueries(1, queryId1);
}

void Graphics::resizeTargets(CameraTargets *targets, Viewport viewport, bool *viewportChanged)
{
    /*int width = camera->getViewport().mWidth;
    int height = camera->getViewport().mHeight;

    glBindTexture(GL_TEXTURE_2D, *colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindTexture(GL_TEXTURE_2D, *depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindTexture(GL_TEXTURE_2D, *colorPickingTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindTexture(GL_TEXTURE_2D, *colorPickingDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    if (camera->mSSAO == CameraSSAO::SSAO_On) {
        glBindTexture(GL_TEXTURE_2D, *positionTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindTexture(GL_TEXTURE_2D, *normalTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindTexture(GL_TEXTURE_2D, *albedoSpecTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindTexture(GL_TEXTURE_2D, *ssaoColorTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    *viewportChanged = false;*/
}

void Graphics::readColorPickingPixel(const CameraTargets *targets, int x, int y, Color32 *color)
{
    glBindFramebuffer(GL_FRAMEBUFFER, targets->mColorPickingFBO);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, color);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createTargets(LightTargets *targets, ShadowMapResolution resolution)
{
    GLsizei res = 512;
    switch (resolution)
    {
    case ShadowMapResolution::Low512x512:
        res = 512;
        break;
    case ShadowMapResolution::Medium1024x1024:
        res = 1024;
        break;
    case ShadowMapResolution::High2048x2048:
        res = 2048;
        break;
    case ShadowMapResolution::VeryHigh4096x4096:
        res = 4096;
        break;
    default:
        res = 1024;
        break;
    }

    // generate shadow map fbos
    // create directional light cascade shadow map fbo
    glGenFramebuffers(5, &(targets->mShadowCascadeFBO[0]));
    glGenTextures(5, &(targets->mShadowCascadeDepthTex[0]));

    for (int i = 0; i < 5; i++)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, targets->mShadowCascadeFBO[i]);
        glBindTexture(GL_TEXTURE_2D, targets->mShadowCascadeDepthTex[i]);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, targets->mShadowCascadeDepthTex[i],
                               0);

        Graphics::checkFrambufferError(__LINE__, __FILE__);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // create spotlight shadow map fbo
    glGenFramebuffers(1, &(targets->mShadowSpotlightFBO));
    glGenTextures(1, &(targets->mShadowSpotlightDepthTex));

    glBindFramebuffer(GL_FRAMEBUFFER, targets->mShadowSpotlightFBO);
    glBindTexture(GL_TEXTURE_2D, targets->mShadowSpotlightDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, targets->mShadowSpotlightDepthTex, 0);

    Graphics::checkFrambufferError(__LINE__, __FILE__);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // create pointlight shadow cubemap fbo
    glGenFramebuffers(1, &(targets->mShadowCubemapFBO));
    glGenTextures(1, &(targets->mShadowCubemapDepthTex));

    glBindFramebuffer(GL_FRAMEBUFFER, targets->mShadowCubemapFBO);
    glBindTexture(GL_TEXTURE_CUBE_MAP, targets->mShadowCubemapDepthTex);

    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 0, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 NULL);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 1, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 NULL);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 2, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 NULL);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 3, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 NULL);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 4, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 NULL);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 5, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 NULL);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, targets->mShadowCubemapDepthTex, 0);

    Graphics::checkFrambufferError(__LINE__, __FILE__);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::destroyTargets(LightTargets *targets)
{
    // detach textures from their framebuffer
    for (int i = 0; i < 5; i++)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, targets->mShadowCascadeFBO[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, targets->mShadowSpotlightFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, targets->mShadowCubemapFBO);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // delete frambuffers
    for (int i = 0; i < 5; i++)
    {
        glDeleteFramebuffers(1, &(targets->mShadowCascadeFBO[i]));
    }
    glDeleteFramebuffers(1, &(targets->mShadowSpotlightFBO));
    glDeleteFramebuffers(1, &(targets->mShadowCubemapFBO));

    // delete textures
    for (int i = 0; i < 5; i++)
    {
        glDeleteTextures(1, &(targets->mShadowCascadeDepthTex[i]));
    }
    glDeleteTextures(1, &(targets->mShadowSpotlightDepthTex));
    glDeleteTextures(1, &(targets->mShadowCubemapDepthTex));

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::resizeTargets(LightTargets *targets, ShadowMapResolution resolution)
{
    GLsizei res = 512;
    switch (resolution)
    {
    case ShadowMapResolution::Low512x512:
        res = 512;
        break;
    case ShadowMapResolution::Medium1024x1024:
        res = 1024;
        break;
    case ShadowMapResolution::High2048x2048:
        res = 2048;
        break;
    case ShadowMapResolution::VeryHigh4096x4096:
        res = 4096;
        break;
    default:
        res = 1024;
        break;
    }

    // If resolution not actually changed, return
    /*if (res == static_cast<GLsizei>(resolution)) {
        *resolutionChanged = false;
        return;
    }*/

    for (int i = 0; i < 5; i++)
    {
        glBindTexture(GL_TEXTURE_2D, targets->mShadowCascadeDepthTex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    glBindTexture(GL_TEXTURE_2D, targets->mShadowSpotlightDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindTexture(GL_TEXTURE_CUBE_MAP, targets->mShadowCubemapDepthTex);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 0, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 NULL);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 1, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 NULL);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 2, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 NULL);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 3, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 NULL);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 4, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 NULL);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 5, 0, GL_DEPTH_COMPONENT, res, res, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createTexture2D(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width, int height, const std::vector<unsigned char> &data,
                               GLuint *tex)
{
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_2D, *tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);
    GLint openglWrapMode = Graphics::getTextureWrapMode(wrapMode);
    GLint openglFilterMode = Graphics::getTextureFilterMode(filterMode);

    glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &data[0]);

    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, openglFilterMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, openglWrapMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, openglWrapMode);

    float aniso = 0.0f;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso);

    // to set aniso
    //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, an);

    Log::info(("Aniso: " + std::to_string(aniso) + "\n").c_str());

    glBindTexture(GL_TEXTURE_2D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::destroyTexture2D(GLuint *tex)
{
    glDeleteTextures(1, tex);
}

void Graphics::updateTexture2D(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, GLuint tex)
{
    GLint openglWrapMode = Graphics::getTextureWrapMode(wrapMode);
    GLint openglFilterMode = Graphics::getTextureFilterMode(filterMode);

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, openglFilterMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, openglWrapMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, openglWrapMode);
    //float aniso = 0.0f;
    //glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso);
    glBindTexture(GL_TEXTURE_2D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::readPixelsTexture2D(TextureFormat format, int width, int height, int numChannels,
                                   std::vector<unsigned char> &data, GLuint tex)
{
    glBindTexture(GL_TEXTURE_2D, tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);

    glGetTextureImage(tex, 0, openglFormat, GL_UNSIGNED_BYTE, width * height * numChannels, &data[0]);

    glBindTexture(GL_TEXTURE_2D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::writePixelsTexture2D(TextureFormat format, int width, int height, const std::vector<unsigned char> &data,
                                    GLuint tex)
{
    glBindTexture(GL_TEXTURE_2D, tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);

    glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &data[0]);

    glBindTexture(GL_TEXTURE_2D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createTexture3D(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width, int height, int depth,
                               const std::vector<unsigned char> &data, GLuint *tex)
{
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_3D, *tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);
    GLint openglWrapMode = Graphics::getTextureWrapMode(wrapMode);
    GLint openglFilterMode = Graphics::getTextureFilterMode(filterMode);

    glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE, &data[0]);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, openglFilterMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, openglWrapMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, openglWrapMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, openglWrapMode);

    glBindTexture(GL_TEXTURE_3D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::destroyTexture3D(GLuint *tex)
{
    glDeleteTextures(1, tex);
}

void Graphics::updateTexture3D(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, GLuint tex)
{
    GLint openglWrapMode = Graphics::getTextureWrapMode(wrapMode);
    GLint openglFilterMode = Graphics::getTextureFilterMode(filterMode);

    glBindTexture(GL_TEXTURE_3D, tex);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, openglFilterMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, openglWrapMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, openglWrapMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, openglWrapMode);
    // to set aniso
    //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, an);
    glBindTexture(GL_TEXTURE_3D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::readPixelsTexture3D(TextureFormat format, int width, int height, int depth, int numChannels,
                                   std::vector<unsigned char> &data, GLuint tex)
{
    glBindTexture(GL_TEXTURE_3D, tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);

    glGetTextureImage(tex, 0, openglFormat, GL_UNSIGNED_BYTE, width * height * depth * numChannels, &data[0]);

    glBindTexture(GL_TEXTURE_3D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::writePixelsTexture3D(TextureFormat format, int width, int height, int depth,
                                    const std::vector<unsigned char> &data, GLuint tex)
{
    glBindTexture(GL_TEXTURE_3D, tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);

    glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE, &data[0]);

    glBindTexture(GL_TEXTURE_3D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createCubemap(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width, const std::vector<unsigned char> &data, GLuint *tex)
{
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, *tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);
    GLint openglWrapMode = Graphics::getTextureWrapMode(wrapMode);
    GLint openglFilterMode = Graphics::getTextureFilterMode(filterMode);

    for (unsigned int i = 0; i < 6; i++)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, width, width, 0, openglFormat,
                     GL_UNSIGNED_BYTE, &data[0]);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, openglFilterMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, openglWrapMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, openglWrapMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, openglWrapMode);

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::destroyCubemap(GLuint *tex)
{
    glDeleteTextures(1, tex);
}

void Graphics::updateCubemap(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, GLuint tex)
{
    GLint openglWrapMode = Graphics::getTextureWrapMode(wrapMode);
    GLint openglFilterMode = Graphics::getTextureFilterMode(filterMode);

    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, openglFilterMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, openglWrapMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, openglWrapMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, openglWrapMode);
    // to set aniso
    //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, an);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::readPixelsCubemap(TextureFormat format, int width, int numChannels, std::vector<unsigned char> &data,
                                 GLuint tex)
{
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);

    for (unsigned int i = 0; i < 6; i++)
    {
        glGetTexImage(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, GL_UNSIGNED_BYTE,
                      &data[i * width * width * numChannels]);
    }

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::writePixelsCubemap(TextureFormat format, int width, const std::vector<unsigned char> &data, GLuint tex)
{
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);

    for (unsigned int i = 0; i < 6; i++)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, width, width, 0, openglFormat,
                     GL_UNSIGNED_BYTE, &data[0]);
    }

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createMesh(const std::vector<float> &vertices, const std::vector<float> &normals,
                          const std::vector<float> &texCoords, GLuint *vao, GLuint *vbo0, GLuint *vbo1, GLuint *vbo2)
{
    glGenVertexArrays(1, vao);
    glBindVertexArray(*vao);
    glGenBuffers(1, vbo0);
    glGenBuffers(1, vbo1);
    glGenBuffers(1, vbo2);

    glBindVertexArray(*vao);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo0);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

    glBindBuffer(GL_ARRAY_BUFFER, *vbo1);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), &normals[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

    glBindBuffer(GL_ARRAY_BUFFER, *vbo2);
    glBufferData(GL_ARRAY_BUFFER, texCoords.size() * sizeof(float), &(texCoords[0]), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

    glBindVertexArray(0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::destroyMesh(GLuint *vao, GLuint *vbo0, GLuint *vbo1, GLuint *vbo2)
{
    glDeleteBuffers(1, vbo0);
    glDeleteBuffers(1, vbo1);
    glDeleteBuffers(1, vbo2);

    glDeleteVertexArrays(1, vao);
}

bool Graphics::compile(const std::string &vert, const std::string &frag, const std::string &geom, GLuint *program)
{
    const GLchar *vertexShaderCharPtr = vert.c_str();
    const GLchar *geometryShaderCharPtr = geom.c_str();
    const GLchar *fragmentShaderCharPtr = frag.c_str();

    GLuint vertexShaderObj = 0;
    GLuint fragmentShaderObj = 0;
    GLuint geometryShaderObj = 0;
    GLint success = 0;
    GLchar infoLog[512];

    // Compile vertex shader
    vertexShaderObj = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShaderObj, 1, &vertexShaderCharPtr, NULL);
    glCompileShader(vertexShaderObj);
    glGetShaderiv(vertexShaderObj, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShaderObj, 512, NULL, infoLog);
        std::string message = "Shader: Vertex shader compilation failed\n";
        Log::error(message.c_str());
    }

    // Compile fragment shader
    fragmentShaderObj = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShaderObj, 1, &fragmentShaderCharPtr, NULL);
    glCompileShader(fragmentShaderObj);
    glGetShaderiv(fragmentShaderObj, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShaderObj, 512, NULL, infoLog);
        std::string message = "Shader: Fragment shader compilation failed\n";
        Log::error(message.c_str());
    }

    // Compile geometry shader
    if (!geom.empty())
    {
        geometryShaderObj = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometryShaderObj, 1, &geometryShaderCharPtr, NULL);
        glCompileShader(geometryShaderObj);
        glGetShaderiv(geometryShaderObj, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(geometryShaderObj, 512, NULL, infoLog);
            std::string message = "Shader: Geometry shader compilation failed\n";
            Log::error(message.c_str());
        }
    }

    // Create shader program
    *program = glCreateProgram();

    // Attach shader objects to shader program
    glAttachShader(*program, vertexShaderObj);
    glAttachShader(*program, fragmentShaderObj);
    if (geometryShaderObj != 0)
    {
        glAttachShader(*program, geometryShaderObj);
    }

    // Link shader program
    glLinkProgram(*program);
    glGetProgramiv(*program, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(*program, 512, NULL, infoLog);
        std::string message = "Shader: Shader program linking failed\n";
        Log::error(message.c_str());
    }

    // Detach shader objects from shader program
    glDetachShader(*program, vertexShaderObj);
    glDetachShader(*program, fragmentShaderObj);
    if (geometryShaderObj != 0)
    {
        glDetachShader(*program, geometryShaderObj);
    }

    // Delete shader objects
    glDeleteShader(vertexShaderObj);
    glDeleteShader(fragmentShaderObj);
    if (!geom.empty())
    {
        glDeleteShader(geometryShaderObj);
    }

    return true;
}

int Graphics::findUniformLocation(const char *name, int program)
{
    return glGetUniformLocation(program, name);
}

int Graphics::getUniformCount(int program)
{
    GLint uniformCount;
    glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &uniformCount);

    return uniformCount;
}

int Graphics::getAttributeCount(int program)
{
    GLint attributeCount;
    glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &attributeCount);

    return attributeCount;
}

std::vector<Uniform> Graphics::getUniforms(int program)
{
    GLint uniformCount;
    glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &uniformCount);

    std::vector<Uniform> uniforms(uniformCount);

    for (int j = 0; j < uniformCount; j++)
    {
        glGetActiveUniform(program, (GLuint)j, 32, &uniforms[j].nameLength, &uniforms[j].size, &uniforms[j].type,
                           &uniforms[j].name[0]);
    }

    return uniforms;
}

std::vector<Attribute> Graphics::getAttributes(int program)
{
    GLint attributeCount;
    glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &attributeCount);

    std::vector<Attribute> attributes(attributeCount);

    for (int j = 0; j < attributeCount; j++)
    {
        glGetActiveAttrib(program, (GLuint)j, 32, &attributes[j].nameLength, &attributes[j].size, &attributes[j].type,
                          &attributes[j].name[0]);
    }

    return attributes;
}

void Graphics::setUniformBlock(const char *blockName, int bindingPoint, int program)
{
    GLuint blockIndex = glGetUniformBlockIndex(program, blockName);
    if (blockIndex != GL_INVALID_INDEX)
    {
        glUniformBlockBinding(program, blockIndex, bindingPoint);
    }
}

void Graphics::use(int program)
{
    glUseProgram(program);
}

void Graphics::unuse()
{
    glUseProgram(0);
}

void Graphics::destroy(int program)
{
    glDeleteProgram(program);
}

void Graphics::setBool(int nameLocation, bool value)
{
    glUniform1i(nameLocation, (int)value);
}
void Graphics::setInt(int nameLocation, int value)
{
    glUniform1i(nameLocation, value);
}

void Graphics::setFloat(int nameLocation, float value)
{
    glUniform1f(nameLocation, value);
}

void Graphics::setColor(int nameLocation, const Color &color)
{
    glUniform4fv(nameLocation, 1, static_cast<const GLfloat *>(&color.r));
}

void Graphics::setVec2(int nameLocation, const glm::vec2 &vec)
{
    glUniform2fv(nameLocation, 1, &vec[0]);
}

void Graphics::setVec3(int nameLocation, const glm::vec3 &vec)
{
    glUniform3fv(nameLocation, 1, &vec[0]);
}

void Graphics::setVec4(int nameLocation, const glm::vec4 &vec)
{
    glUniform4fv(nameLocation, 1, &vec[0]);
}

void Graphics::setMat2(int nameLocation, const glm::mat2 &mat)
{
    glUniformMatrix2fv(nameLocation, 1, GL_FALSE, &mat[0][0]);
}

void Graphics::setMat3(int nameLocation, const glm::mat3 &mat)
{
    glUniformMatrix3fv(nameLocation, 1, GL_FALSE, &mat[0][0]);
}

void Graphics::setMat4(int nameLocation, const glm::mat4 &mat)
{
    glUniformMatrix4fv(nameLocation, 1, GL_FALSE, &mat[0][0]);
}

void Graphics::setTexture2D(int nameLocation, int texUnit, int tex)
{
    glUniform1i(nameLocation, texUnit);

    glActiveTexture(GL_TEXTURE0 + texUnit);
    glBindTexture(GL_TEXTURE_2D, tex);
}

bool Graphics::getBool(int nameLocation, int program)
{
    int value = 0;
    glGetUniformiv(program, nameLocation, &value);

    return (bool)value;
}

int Graphics::getInt(int nameLocation, int program)
{
    int value = 0;
    glGetUniformiv(program, nameLocation, &value);

    return value;
}

float Graphics::getFloat(int nameLocation, int program)
{
    float value = 0.0f;
    glGetUniformfv(program, nameLocation, &value);

    return value;
}

Color Graphics::getColor(int nameLocation, int program)
{
    Color color = Color(0.0f, 0.0f, 0.0f, 1.0f);
    glGetnUniformfv(program, nameLocation, sizeof(Color), &color.r);

    return color;
}

glm::vec2 Graphics::getVec2(int nameLocation, int program)
{
    glm::vec2 value = glm::vec2(0.0f);
    glGetnUniformfv(program, nameLocation, sizeof(glm::vec2), &value[0]);

    return value;
}

glm::vec3 Graphics::getVec3(int nameLocation, int program)
{
    glm::vec3 value = glm::vec3(0.0f);
    glGetnUniformfv(program, nameLocation, sizeof(glm::vec3), &value[0]);

    return value;
}

glm::vec4 Graphics::getVec4(int nameLocation, int program)
{
    glm::vec4 value = glm::vec4(0.0f);
    glGetnUniformfv(program, nameLocation, sizeof(glm::vec4), &value[0]);

    return value;
}

glm::mat2 Graphics::getMat2(int nameLocation, int program)
{
    glm::mat2 value = glm::mat2(0.0f);
    glGetnUniformfv(program, nameLocation, sizeof(glm::mat2), &value[0][0]);

    return value;
}

glm::mat3 Graphics::getMat3(int nameLocation, int program)
{
    glm::mat3 value = glm::mat3(0.0f);
    glGetnUniformfv(program, nameLocation, sizeof(glm::mat3), &value[0][0]);

    return value;
}

glm::mat4 Graphics::getMat4(int nameLocation, int program)
{
    glm::mat4 value = glm::mat4(0.0f);
    glGetnUniformfv(program, nameLocation, sizeof(glm::mat4), &value[0][0]);

    return value;
}

int Graphics::getTexture2D(int nameLocation, int texUnit, int program)
{
    int tex = -1;
    glActiveTexture(GL_TEXTURE0 + texUnit);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex);

    return tex;
}

void Graphics::applyMaterial(const std::vector<ShaderUniform> &uniforms, const std::vector<GLint> &textures,
                             int shaderProgram)
{
    int textureUnit = 0;
    for (size_t i = 0; i < uniforms.size(); i++)
    {
        if (uniforms[i].mType == GL_SAMPLER_2D)
        {
            if (textures[textureUnit] != -1)
            {
                Graphics::setTexture2D(findUniformLocation(uniforms[i].mName, shaderProgram), textureUnit,
                                       textures[textureUnit]);
            }
            else
            {
                Graphics::setTexture2D(findUniformLocation(uniforms[i].mName, shaderProgram), textureUnit, 0);
            }

            textureUnit++;
        }
        else if (uniforms[i].mType == GL_INT)
        {
            Graphics::setInt(findUniformLocation(uniforms[i].mName, shaderProgram),
                             *reinterpret_cast<const int *>(uniforms[i].mData));
        }
        else if (uniforms[i].mType == GL_FLOAT)
        {
            Graphics::setFloat(findUniformLocation(uniforms[i].mName, shaderProgram),
                               *reinterpret_cast<const float *>(uniforms[i].mData));
        }
        else if (uniforms[i].mType == GL_FLOAT_VEC2)
        {
            Graphics::setVec2(findUniformLocation(uniforms[i].mName, shaderProgram),
                              *reinterpret_cast<const glm::vec2 *>(uniforms[i].mData));
        }
        else if (uniforms[i].mType == GL_FLOAT_VEC3)
        {
            Graphics::setVec3(findUniformLocation(uniforms[i].mName, shaderProgram),
                              *reinterpret_cast<const glm::vec3 *>(uniforms[i].mData));
        }
        else if (uniforms[i].mType == GL_FLOAT_VEC4)
        {
            Graphics::setVec4(findUniformLocation(uniforms[i].mName, shaderProgram),
                              *reinterpret_cast<const glm::vec4 *>(uniforms[i].mData));
        }
    }

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::render(int start, int count, GLuint vao, bool wireframe)
{
    if (wireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, start, count);
    glBindVertexArray(0);

    if (wireframe)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::render(const RenderObject &renderObject, GraphicsQuery &query)
{
    GLsizei numVertices = renderObject.size / 3;

    Graphics::render(renderObject.start / 3, numVertices, renderObject.vao);

    query.mNumDrawCalls++;
    query.mVerts += numVertices;
    query.mTris += numVertices / 3;
}