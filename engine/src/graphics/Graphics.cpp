#include <GL/glew.h>
#include <algorithm>
#include <iostream>
#include <random>

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

void Graphics::turnOn(Capability capability)
{
    switch(capability)
    {
    case Capability::Depth_Testing:
        glEnable(GL_DEPTH_TEST);
        break;
    case Capability::Blending:
        glEnable(GL_BLEND);
        break;
    }
}

void Graphics::turnOff(Capability capability)
{
    switch (capability)
    {
    case Capability::Depth_Testing:
        glDisable(GL_DEPTH_TEST);
        break;
    case Capability::Blending:
        glDisable(GL_BLEND);
        break;
    }
}

GLenum Graphics::getTextureFormat(TextureFormat format)
{
    GLenum openglFormat = GL_DEPTH_COMPONENT;

    switch (format)
    {
    case TextureFormat::Depth:
        openglFormat = GL_DEPTH_COMPONENT;
        break;
    case TextureFormat::RG:
        openglFormat = GL_RG;
        break;
    case TextureFormat::RGB:
        openglFormat = GL_RGB;
        break;
    case TextureFormat::RGBA:
        openglFormat = GL_RGBA;
        break;
    default:
        Log::error("OpengGL: Invalid texture format\n");
        break;
    }

    return openglFormat;
}

int Graphics::getTextureWrapMode(TextureWrapMode wrapMode)
{
    int openglWrapMode = GL_REPEAT;

    switch (wrapMode)
    {
    case TextureWrapMode::Repeat:
        openglWrapMode = GL_REPEAT;
        break;
    case TextureWrapMode::Clamp:
        openglWrapMode = GL_CLAMP_TO_EDGE;
        break;
    default:
        Log::error("OpengGL: Invalid texture wrap mode\n");
        break;
    }

    return openglWrapMode;
}

int Graphics::getTextureFilterMode(TextureFilterMode filterMode)
{
    int openglFilterMode = GL_NEAREST;

    switch (filterMode)
    {
    case TextureFilterMode::Nearest:
        openglFilterMode = GL_NEAREST;
        break;
    case TextureFilterMode::Bilinear:
        openglFilterMode = GL_LINEAR;
        break;
    case TextureFilterMode::Trilinear:
        openglFilterMode = GL_LINEAR_MIPMAP_LINEAR;
        break;
    default:
        Log::error("OpengGL: Invalid texture filter mode\n");
        break;
    }

    return openglFilterMode;
}

void Graphics::beginQuery(unsigned int queryId)
{
    glBeginQuery(GL_TIME_ELAPSED, queryId);
}

void Graphics::endQuery(unsigned int queryId, unsigned long long *elapsedTime)
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

    float spotAngle = glm::cos(glm::radians(uniform.mSpotAngle));
    float innerSpotAngle = glm::cos(glm::radians(uniform.mInnerSpotAngle));
    glBufferSubData(GL_UNIFORM_BUFFER, 772, 4, &(spotAngle));
    glBufferSubData(GL_UNIFORM_BUFFER, 776, 4, &(innerSpotAngle));
    glBufferSubData(GL_UNIFORM_BUFFER, 780, 4, &(uniform.mShadowNearPlane));
    glBufferSubData(GL_UNIFORM_BUFFER, 784, 4, &(uniform.mShadowFarPlane));
    glBufferSubData(GL_UNIFORM_BUFFER, 788, 4, &(uniform.mShadowBias));
    glBufferSubData(GL_UNIFORM_BUFFER, 792, 4, &(uniform.mShadowRadius));
    glBufferSubData(GL_UNIFORM_BUFFER, 796, 4, &(uniform.mShadowStrength));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createScreenQuad(unsigned int* vao, unsigned int* vbo)
{
    float quadVertices[] = { // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    glGenVertexArrays(1, vao);
    glGenBuffers(1, vbo);
    glBindVertexArray(*vao);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::renderScreenQuad(unsigned int vao)
{
    glDisable(GL_DEPTH_TEST);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createFramebuffer(int width, int height, unsigned int* fbo, unsigned int* color, unsigned int* depth)
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
    unsigned int mainAttachments[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, mainAttachments);

    Graphics::checkFrambufferError(__LINE__, __FILE__);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Graphics::destroyFramebuffer(unsigned int* fbo, unsigned int* color, unsigned int* depth)
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

void Graphics::bindFramebuffer(unsigned int fbo)
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

void Graphics::bindVertexArray(unsigned int vao)
{
    glBindVertexArray(vao);
}

void Graphics::unbindVertexArray()
{
    glBindVertexArray(0);
}

void Graphics::setViewport(int x, int y, int width, int height)
{
    glViewport(x, y, width, height);
}

void Graphics::createTargets(CameraTargets *targets, Viewport viewport, glm::vec3 *ssaoSamples, unsigned int *queryId0,
                             unsigned int *queryId1)
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




    // create and attach depth buffer (renderbuffer)
    unsigned int rboDepth;
    glGenRenderbuffers(1, &rboDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1920, 1080);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);




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

void Graphics::destroyTargets(CameraTargets *targets, unsigned int *queryId0, unsigned int *queryId1)
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
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

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

void Graphics::createTexture2D(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                               int height, const std::vector<unsigned char> &data, unsigned int *tex)
{
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_2D, *tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);
    GLint openglWrapMode = Graphics::getTextureWrapMode(wrapMode);
    GLint openglFilterMode = Graphics::getTextureFilterMode(filterMode);

    glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, data.data());

    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, openglFilterMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, openglWrapMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, openglWrapMode);

    float aniso = 0.0f;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso);

    // to set aniso
    // glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, an);

    glBindTexture(GL_TEXTURE_2D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::destroyTexture2D(unsigned int *tex)
{
    glDeleteTextures(1, tex);
}

void Graphics::updateTexture2D(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, unsigned int tex)
{
    GLint openglWrapMode = Graphics::getTextureWrapMode(wrapMode);
    GLint openglFilterMode = Graphics::getTextureFilterMode(filterMode);

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, openglFilterMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, openglWrapMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, openglWrapMode);
    // float aniso = 0.0f;
    // glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso);
    glBindTexture(GL_TEXTURE_2D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::readPixelsTexture2D(TextureFormat format, int width, int height, int numChannels,
                                   std::vector<unsigned char> &data, unsigned int tex)
{
    glBindTexture(GL_TEXTURE_2D, tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);

    glGetTextureImage(tex, 0, openglFormat, GL_UNSIGNED_BYTE, width * height * numChannels, &data[0]);

    glBindTexture(GL_TEXTURE_2D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::writePixelsTexture2D(TextureFormat format, int width, int height, const std::vector<unsigned char> &data,
                                    unsigned int tex)
{
    glBindTexture(GL_TEXTURE_2D, tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);

    glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, data.data());

    glBindTexture(GL_TEXTURE_2D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createTexture3D(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                               int height, int depth, const std::vector<unsigned char> &data, unsigned int *tex)
{
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_3D, *tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);
    GLint openglWrapMode = Graphics::getTextureWrapMode(wrapMode);
    GLint openglFilterMode = Graphics::getTextureFilterMode(filterMode);

    glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE, &data[0]);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, openglFilterMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER,
                    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, openglWrapMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, openglWrapMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, openglWrapMode);

    glBindTexture(GL_TEXTURE_3D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::destroyTexture3D(unsigned int *tex)
{
    glDeleteTextures(1, tex);
}

void Graphics::updateTexture3D(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, unsigned int tex)
{
    GLint openglWrapMode = Graphics::getTextureWrapMode(wrapMode);
    GLint openglFilterMode = Graphics::getTextureFilterMode(filterMode);

    glBindTexture(GL_TEXTURE_3D, tex);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, openglFilterMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER,
                    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, openglWrapMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, openglWrapMode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, openglWrapMode);
    // to set aniso
    // glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, an);
    glBindTexture(GL_TEXTURE_3D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::readPixelsTexture3D(TextureFormat format, int width, int height, int depth, int numChannels,
                                   std::vector<unsigned char> &data, unsigned int tex)
{
    glBindTexture(GL_TEXTURE_3D, tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);

    glGetTextureImage(tex, 0, openglFormat, GL_UNSIGNED_BYTE, width * height * depth * numChannels, &data[0]);

    glBindTexture(GL_TEXTURE_3D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::writePixelsTexture3D(TextureFormat format, int width, int height, int depth,
                                    const std::vector<unsigned char> &data, unsigned int tex)
{
    glBindTexture(GL_TEXTURE_3D, tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);

    glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE, &data[0]);

    glBindTexture(GL_TEXTURE_3D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createCubemap(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                             const std::vector<unsigned char> &data, unsigned int *tex)
{
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, *tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);
    GLint openglWrapMode = Graphics::getTextureWrapMode(wrapMode);
    GLint openglFilterMode = Graphics::getTextureFilterMode(filterMode);

    for (unsigned int i = 0; i < 6; i++)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, width, width, 0, openglFormat,
                     GL_UNSIGNED_BYTE, data.data());
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, openglFilterMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER,
                    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, openglWrapMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, openglWrapMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, openglWrapMode);

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::destroyCubemap(unsigned int *tex)
{
    glDeleteTextures(1, tex);
}

void Graphics::updateCubemap(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, unsigned int tex)
{
    GLint openglWrapMode = Graphics::getTextureWrapMode(wrapMode);
    GLint openglFilterMode = Graphics::getTextureFilterMode(filterMode);

    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, openglFilterMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER,
                    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, openglWrapMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, openglWrapMode);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, openglWrapMode);
    // to set aniso
    // glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, an);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::readPixelsCubemap(TextureFormat format, int width, int numChannels, std::vector<unsigned char> &data,
                                 unsigned int tex)
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

void Graphics::writePixelsCubemap(TextureFormat format, int width, const std::vector<unsigned char> &data, unsigned int tex)
{
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);

    for (unsigned int i = 0; i < 6; i++)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, width, width, 0, openglFormat,
                     GL_UNSIGNED_BYTE, data.data());
    }

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createRenderTextureTargets(RenderTextureTargets* targets, TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width, int height)
{
    // generate fbo (color + depth)
    glGenFramebuffers(1, &(targets->mMainFBO));
    glBindFramebuffer(GL_FRAMEBUFFER, targets->mMainFBO);

    glGenTextures(1, &(targets->mColorTex));
    glBindTexture(GL_TEXTURE_2D, targets->mColorTex);
    
    GLenum openglFormat = Graphics::getTextureFormat(format);
    GLint openglWrapMode = Graphics::getTextureWrapMode(wrapMode);
    GLint openglFilterMode = Graphics::getTextureFilterMode(filterMode);
    
    glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    //glGenerateMipmap(GL_TEXTURE_2D);

    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, openglFilterMode);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
    //    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, openglWrapMode);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, openglWrapMode);

    //float aniso = 0.0f;
    //glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso);

    glGenTextures(1, &(targets->mDepthTex));
    glBindTexture(GL_TEXTURE_2D, targets->mDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, openglFilterMode);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
    //    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, openglWrapMode);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, openglWrapMode);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, targets->mColorTex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, targets->mDepthTex, 0);

    // - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
    unsigned int mainAttachments[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, mainAttachments);

    Graphics::checkFrambufferError(__LINE__, __FILE__);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Graphics::destroyRenderTextureTargets(RenderTextureTargets* targets)
{
    // detach textures from their framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, targets->mMainFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // delete frambuffers
    glDeleteFramebuffers(1, &(targets->mMainFBO));

    // delete textures
    glDeleteTextures(1, &(targets->mColorTex));
    glDeleteTextures(1, &(targets->mDepthTex));
}

void Graphics::createMesh(const std::vector<float> &vertices, const std::vector<float> &normals,
                          const std::vector<float> &texCoords, unsigned int*vao, unsigned int*vbo0, unsigned int*vbo1, unsigned int*vbo2)
{
    glGenVertexArrays(1, vao);
    glBindVertexArray(*vao);
    glGenBuffers(1, vbo0);
    glGenBuffers(1, vbo1);
    glGenBuffers(1, vbo2);

    glBindVertexArray(*vao);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo0);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

    glBindBuffer(GL_ARRAY_BUFFER, *vbo1);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

    glBindBuffer(GL_ARRAY_BUFFER, *vbo2);
    glBufferData(GL_ARRAY_BUFFER, texCoords.size() * sizeof(float), texCoords.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

    glBindVertexArray(0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::destroyMesh(unsigned int*vao, unsigned int*vbo0, unsigned int*vbo1, unsigned int*vbo2)
{
    glDeleteBuffers(1, vbo0);
    glDeleteBuffers(1, vbo1);
    glDeleteBuffers(1, vbo2);

    glDeleteVertexArrays(1, vao);
}

void Graphics::createSprite(unsigned int* vao)
{
    // configure VAO/VBO
    unsigned int vbo;
    float vertices[] = {
        // pos      // tex
        0.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,

        0.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 0.0f, 1.0f, 0.0f
    };

    glGenVertexArrays(1, vao);
    glGenBuffers(1, &vbo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindVertexArray(*vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Graphics::destroySprite(unsigned int* vao)
{
    glDeleteVertexArrays(1, vao);
}

void Graphics::preprocess(std::string& vert, std::string& frag, std::string& geom, int64_t variant)
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

    //pos = geom.find('\n');
    //if (pos != std::string::npos)
    //{
    //    version = geom.substr(0, pos + 1);
    //    shader = geom.substr(pos + 1);
    //}

    //geom = version + defines + shader;
}

bool Graphics::compile(const std::string &name, const std::string &vert, const std::string &frag, const std::string &geom, unsigned int *program)
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
        std::string message = "Shader: Vertex shader compilation failed (" + name + ")\n";
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
        std::string message = "Shader: Fragment shader compilation failed (" + name + ")\n";
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
            std::string message = "Shader: Geometry shader compilation failed (" + name + ")\n";
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
        std::string message = "Shader: " + name + " program linking failed\n";
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

std::vector<ShaderUniform> Graphics::getShaderUniforms(int program)
{
    GLint uniformCount;
    glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &uniformCount);

    std::vector<ShaderUniform> uniforms(uniformCount);

    for (int j = 0; j < uniformCount; j++)
    {
        Uniform uniform;
        glGetActiveUniform(program, (GLuint)j, 32, &uniform.nameLength, &uniform.size, &uniform.type,
            &uniform.name[0]);

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
    }

    return uniforms;
}

std::vector<ShaderAttribute> Graphics::getShaderAttributes(int program)
{
    GLint attributeCount;
    glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &attributeCount);

    std::vector<ShaderAttribute> attributes(attributeCount);

    for (int j = 0; j < attributeCount; j++)
    {
        Attribute attrib;
        glGetActiveAttrib(program, (GLuint)j, 32, &attrib.nameLength, &attrib.size, &attrib.type,
                          &attrib.name[0]);

        attributes[j].mName = std::string(attrib.name);
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

void Graphics::setColor32(int nameLocation, const Color32& color)
{
    glUniform4ui(nameLocation, static_cast<GLuint>(color.r), 
        static_cast<GLuint>(color.g), static_cast<GLuint>(color.b), static_cast<GLuint>(color.a));
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

Color32 Graphics::getColor32(int nameLocation, int program)
{
    Color32 color = Color32(0, 0, 0, 255);
    
    GLuint c[4];
    glGetnUniformuiv(program,
        nameLocation,
        4 * sizeof(GLuint),
        &c[0]);

    color.r = static_cast<unsigned char>(c[0]);
    color.g = static_cast<unsigned char>(c[1]);
    color.b = static_cast<unsigned char>(c[2]);
    color.a = static_cast<unsigned char>(c[3]);
    
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

void Graphics::applyMaterial(const std::vector<ShaderUniform> &uniforms, const std::vector<int> &textures,
                             int shaderProgram)
{
    int textureUnit = 0;
    for (size_t i = 0; i < uniforms.size(); i++)
    {
        if (uniforms[i].mType == ShaderUniformType::Sampler2D)
        {
            if (textures[textureUnit] != -1)
            {
                Graphics::setTexture2D(findUniformLocation(uniforms[i].mName.c_str(), shaderProgram), textureUnit,
                                       textures[textureUnit]);
            }
            else
            {
                Graphics::setTexture2D(findUniformLocation(uniforms[i].mName.c_str(), shaderProgram), textureUnit, 0);
            }

            textureUnit++;
        }
        else if (uniforms[i].mType == ShaderUniformType::Int)
        {
            Graphics::setInt(findUniformLocation(uniforms[i].mName.c_str(), shaderProgram),
                             *reinterpret_cast<const int *>(uniforms[i].mData));
        }
        else if (uniforms[i].mType == ShaderUniformType::Float)
        {
            Graphics::setFloat(findUniformLocation(uniforms[i].mName.c_str(), shaderProgram),
                               *reinterpret_cast<const float *>(uniforms[i].mData));
        }
        else if (uniforms[i].mType == ShaderUniformType::Vec2)
        {
            Graphics::setVec2(findUniformLocation(uniforms[i].mName.c_str(), shaderProgram),
                              *reinterpret_cast<const glm::vec2 *>(uniforms[i].mData));
        }
        else if (uniforms[i].mType == ShaderUniformType::Vec3)
        {
            Graphics::setVec3(findUniformLocation(uniforms[i].mName.c_str(), shaderProgram),
                              *reinterpret_cast<const glm::vec3 *>(uniforms[i].mData));
        }
        else if (uniforms[i].mType == ShaderUniformType::Vec4)
        {
            Graphics::setVec4(findUniformLocation(uniforms[i].mName.c_str(), shaderProgram),
                              *reinterpret_cast<const glm::vec4 *>(uniforms[i].mData));
        }
    }

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::render(int start, int count, int vao, bool wireframe)
{
    if (wireframe)
    {
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
    int numVertices = renderObject.size / 3;

    Graphics::render(renderObject.start / 3, numVertices, renderObject.vao);

    query.mNumDrawCalls++;
    query.mVerts += numVertices;
    query.mTris += numVertices / 3;
}

void Graphics::compileSSAOShader(ForwardRendererState &state)
{
    std::string vertexShader0 = "#version 430 core\n"
                               "layout(std140) uniform CameraBlock\n"
                               "{\n"
                               "    mat4 projection;\n"
                               "    mat4 view;\n"
                               "    vec3 cameraPos;\n"
                               "}Camera;\n"
                               "in vec3 position;\n"
                               "in vec3 normal;\n"
                               "in vec2 texCoord;\n"
                               "out vec3 FragPos;\n"
                               "out vec3 Normal;\n"
                               "uniform mat4 model;\n"
                               "void main()\n"
                               "{\n"
                               "    vec4 viewPos = Camera.view * model * vec4(position, 1.0);\n"
                               "    FragPos = viewPos.xyz;\n"
                               "    mat3 normalMatrix = transpose(inverse(mat3(Camera.view * model)));\n"
                               "    Normal = normalMatrix * normal;\n"
                               "    gl_Position = Camera.projection * viewPos;\n"
                               "}\n";

    std::string fragmentShader0 = "#version 430 core\n"
                                 "layout(location = 0) out vec3 positionTex;\n"
                                 "layout(location = 1) out vec3 normalTex;\n"
                                 "in vec3 FragPos;\n"
                                 "in vec3 Normal;\n"
                                 "void main()\n"
                                 "{\n"
                                 "   // store the fragment position vector in the first gbuffer texture\n"
                                 "   positionTex = FragPos.xyz;\n"
                                 "   // also store the per-fragment normals into the gbuffer\n"
                                 "   normalTex = normalize(Normal);\n"
                                 "}\n";

    unsigned int program = 0;
    if (Graphics::compile("Geometry", vertexShader0, fragmentShader0, "", &program))
    {
        state.mGeometryShaderProgram = program;
        state.mGeometryShaderModelLoc = Graphics::findUniformLocation("model", state.mGeometryShaderProgram);

        Graphics::setUniformBlock("CameraBlock", 0, state.mGeometryShaderProgram);
    }
    else
    {
        state.mGeometryShaderProgram = -1;
    }

    std::string vertexShader1 = "#version 430 core\n"
                                "in vec3 position;\n"
                                "in vec2 texCoord;\n"
                                "out vec2 TexCoord;\n"
                                "void main()\n"
                                "{\n"
                                "   gl_Position = vec4(position, 1.0);\n"
                                "   TexCoord = texCoord;\n"
                                "}\n";

    std::string fragmentShader1 =
        "#version 430 core\n"
        "out float FragColor;\n"
        "in vec2 TexCoord;\n"
        "uniform sampler2D positionTex;\n"
        "uniform sampler2D normalTex;\n"
        "uniform sampler2D noiseTex;\n"
        "uniform vec3 samples[64];\n"
        "// parameters (you'd probably want to use them as uniforms to more easily tweak the effect)\n"
        "int kernelSize = 64;\n"
        "float radius = 0.5;\n"
        "float bias = 0.025;\n"
        "// tile noise texture over screen based on screen dimensions divided by noise size\n"
        "const vec2 noiseScale = vec2(1024.0 / 4.0, 1024.0 / 4.0);\n"
        "uniform mat4 projection;\n"
        "void main()\n"
        "{\n"
        "   // get input for SSAO algorithm\n"
        "   vec3 fragPos = texture(positionTex, TexCoord).xyz;\n"
        "   vec3 normal = normalize(texture(normalTex, TexCoord).rgb);\n"
        "   vec3 randomVec = normalize(texture(noiseTex, TexCoord * noiseScale).xyz);\n"
        "   // create TBN change-of-basis matrix: from tangent-space to view-space\n"
        "   vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));\n"
        "   vec3 bitangent = cross(normal, tangent);\n"
        "   mat3 TBN = mat3(tangent, bitangent, normal);\n"
        "   // iterate over the sample kernel and calculate occlusion factor\n"
        "   float occlusion = 0.0f;\n"
        "   for (int i = 0; i < kernelSize; ++i)\n"
        "   {\n"
        "       // get sample position\n"
        "       vec3 sampleq = TBN * samples[i]; // from tangent to view-space\n"
        "       sampleq = fragPos + sampleq * radius;\n"
        "       // project sample position (to sample texture) (to get position on screen/texture)\n"
        "       vec4 offset = vec4(sampleq, 1.0);\n"
        "       offset = projection * offset; // from view to clip-space\n"
        "       offset.xyz /= offset.w;       // perspective divide\n"
        "       offset.xyz = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0\n"
        "       // get sample depth\n"
        "       float sampleDepth = texture(positionTex, offset.xy).z; // get depth value of kernel sample\n"
        "       // range check & accumulate\n"
        "       float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));\n"
        "       occlusion += (sampleDepth >= sampleq.z + bias ? 1.0 : 0.0) * rangeCheck;\n"
        "   }\n"
        "   occlusion = 1.0 - (occlusion / kernelSize);\n"
        "   FragColor = occlusion;\n"
        "}\n";

    program = 0;
    if (Graphics::compile("SSAO", vertexShader1, fragmentShader1, "", &program))
    {
        state.mSsaoShaderProgram = program;
        state.mSsaoShaderProjectionLoc = Graphics::findUniformLocation("projection", state.mSsaoShaderProgram);
        state.mSsaoShaderPositionTexLoc = Graphics::findUniformLocation("positionTex", state.mSsaoShaderProgram);
        state.mSsaoShaderNormalTexLoc = Graphics::findUniformLocation("normalTex", state.mSsaoShaderProgram);
        state.mSsaoShaderNoiseTexLoc = Graphics::findUniformLocation("noiseTex", state.mSsaoShaderProgram);

        for (int i = 0; i < 64; i++)
        {
            std::string sample = "samples[" + std::to_string(i) + "]";
            state.mSsaoShaderSamplesLoc[i] = Graphics::findUniformLocation(sample.c_str(), state.mSsaoShaderProgram);
        }
    }
    else
    {
        state.mSsaoShaderProgram = -1;
    }
}

void Graphics::compileShadowDepthMapShader(ForwardRendererState &state)
{
    std::string vertexShader = "#version 430 core\n"
                               "uniform mat4 projection;\n"
                               "uniform mat4 view;\n"
                               "uniform mat4 model;\n"
                               "in vec3 position;\n"
                               "void main()\n"
                               "{\n"
                               "    gl_Position = projection * view * model * vec4(position, 1.0);\n"
                               "}\n";

    std::string fragmentShader = "#version 430 core\n"
                                 "void main()\n"
                                 "{\n"
                                 "}\n";

    unsigned int program = 0;
    if (Graphics::compile("Shadow Depth Map", vertexShader, fragmentShader, "", &program))
    {
        state.mDepthShaderProgram = program;
        state.mDepthShaderModelLoc = Graphics::findUniformLocation("model", state.mDepthShaderProgram);
        state.mDepthShaderViewLoc = Graphics::findUniformLocation("view", state.mDepthShaderProgram);
        state.mDepthShaderProjectionLoc = Graphics::findUniformLocation("projection", state.mDepthShaderProgram);
    }
    else
    {
        state.mDepthShaderProgram = -1;
    }
}

void Graphics::compileShadowDepthCubemapShader(ForwardRendererState &state)
{
    std::string vertexShader = "#version 430 core\n"
                               "in vec3 position;\n"
                               "uniform mat4 model;\n"
                               "void main()\n"
                               "{\n"
                               "    gl_Position = model * vec4(position, 1.0);\n"
                               "}\n";

    std::string fragmentShader = "#version 430 core\n"
                                 "in vec4 FragPos;\n"
                                 "uniform vec3 lightPos;\n"
                                 "uniform float farPlane;\n"
                                 "void main()\n"
                                 "{\n"
                                 "  float lightDistance = length(FragPos.xyz - lightPos);\n"
                                 "  lightDistance = lightDistance / farPlane;\n"
                                 "  gl_FragDepth = lightDistance;\n"
                                 "}\n";

    std::string geometryShader = "#version 430 core\n"
                                 "layout(triangles) in;\n"
                                 "layout(triangle_strip, max_vertices = 18) out;\n"
                                 "uniform mat4 cubeViewProjMatrices[6];\n"
                                 "out vec4 FragPos;\n"
                                 "void main()\n"
                                 "{\n"
                                 "  for (int i = 0; i < 6; i++)\n"
                                 "  {\n"
                                 "      gl_Layer = i;\n"
                                 "      for (int j = 0; j < 3; j++)\n"
                                 "      {\n"
                                 "          FragPos = gl_in[j].gl_Position;\n"
                                 "          gl_Position = cubeViewProjMatrices[i] * FragPos;\n"
                                 "          EmitVertex();\n"
                                 "      }\n"
                                 "      EndPrimitive();\n"
                                 "  }\n"
                                 "}\n";

    unsigned int program = 0;
    if (Graphics::compile("Shadow Depth Cubemap", vertexShader, fragmentShader, geometryShader, &program))
    {
        state.mDepthCubemapShaderProgram = program;
        state.mDepthCubemapShaderLightPosLoc =
            Graphics::findUniformLocation("lightPos", state.mDepthCubemapShaderProgram);
        state.mDepthCubemapShaderFarPlaneLoc =
            Graphics::findUniformLocation("farPlane", state.mDepthCubemapShaderProgram);
        state.mDepthCubemapShaderModelLoc = Graphics::findUniformLocation("model", state.mDepthCubemapShaderProgram);
        state.mDepthCubemapShaderCubeViewProjMatricesLoc0 =
            Graphics::findUniformLocation("cubeViewProjMatrices[0]", state.mDepthCubemapShaderProgram);
        state.mDepthCubemapShaderCubeViewProjMatricesLoc1 =
            Graphics::findUniformLocation("cubeViewProjMatrices[1]", state.mDepthCubemapShaderProgram);
        state.mDepthCubemapShaderCubeViewProjMatricesLoc2 =
            Graphics::findUniformLocation("cubeViewProjMatrices[2]", state.mDepthCubemapShaderProgram);
        state.mDepthCubemapShaderCubeViewProjMatricesLoc3 =
            Graphics::findUniformLocation("cubeViewProjMatrices[3]", state.mDepthCubemapShaderProgram);
        state.mDepthCubemapShaderCubeViewProjMatricesLoc4 =
            Graphics::findUniformLocation("cubeViewProjMatrices[4]", state.mDepthCubemapShaderProgram);
        state.mDepthCubemapShaderCubeViewProjMatricesLoc5 =
            Graphics::findUniformLocation("cubeViewProjMatrices[5]", state.mDepthCubemapShaderProgram);
    }
    else
    {
        state.mDepthCubemapShaderProgram = -1;
    }
}

void Graphics::compileColorShader(ForwardRendererState &state)
{
    std::string vertexShader = "#version 430 core\n"
                               "layout(std140) uniform CameraBlock\n"
                               "{\n"
                               "    mat4 projection;\n"
                               "    mat4 view;\n"
                               "    vec3 cameraPos;\n"
                               "}Camera;\n"
                               "uniform mat4 model;\n"
                               "in vec3 position;\n"
                               "void main()\n"
                               "{\n"
                               "    gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);\n"
                               "}\n";

    std::string fragmentShader = "#version 430 core\n"
                                 "struct Material\n"
                                 "{\n"
                                 "    uvec4 color;\n"
                                 "};\n"
                                 "uniform Material material;\n"
                                 "out vec4 FragColor;\n"
                                 "void main()\n"
                                 "{\n"
                                 "    FragColor = vec4(material.color.r / 255.0f, material.color.g / 255.0f,\n"
                                 "                      material.color.b / 255.0f, material.color.a / 255.0f);\n"
                                 "}\n";

    unsigned int program = 0;
    if (Graphics::compile("Color", vertexShader, fragmentShader, "", &program))
    {
        state.mColorShaderProgram = program;
        state.mColorShaderModelLoc = Graphics::findUniformLocation("model", state.mColorShaderProgram);
        state.mColorShaderColorLoc = Graphics::findUniformLocation("material.color", state.mColorShaderProgram);

        Graphics::setUniformBlock("CameraBlock", 0, state.mColorShaderProgram);
    }
    else
    {
        state.mColorShaderProgram = -1;
    }
}

void Graphics::compileScreenQuadShader(ForwardRendererState &state)
{
    std::string vertexShader = "#version 330 core\n"
                               "layout(location = 0) in vec2 aPos;\n"
                               "layout(location = 1) in vec2 aTexCoords;\n"
                               "out vec2 TexCoords;\n"
                               "void main()\n"
                               "{\n"
                               "    TexCoords = aTexCoords;\n"
                               "    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);\n"
                               "}\n";

    std::string fragmentShader = "#version 330 core\n"
                                 "out vec4 FragColor;\n"
                                 "in vec2 TexCoords;\n"
                                 "uniform sampler2D screenTexture;\n"
                                 "void main()\n"
                                 "{\n"
                                 "  vec3 col = texture(screenTexture, TexCoords).rgb;\n"
                                 "  FragColor = vec4(col, 1.0);\n"
                                 "}\n";

    unsigned int program = 0;
    if (Graphics::compile("Screen Quad", vertexShader, fragmentShader, "", &program))
    {
        state.mQuadShaderProgram = program;
        state.mQuadShaderTexLoc = Graphics::findUniformLocation("screenTexture", state.mQuadShaderProgram);
    }
    else
    {
        state.mQuadShaderProgram = -1;
    }
}

void Graphics::compileSpriteShader(ForwardRendererState &state)
{
    std::string vertexShader = "#version 430 core\n"
                               "layout(location = 0) in vec4 vertex; // <vec2 position, vec2 texCoords>\n"
                               "out vec2 TexCoords;\n"
                               "uniform mat4 model;\n"
                               "uniform mat4 view;\n"
                               "uniform mat4 projection;\n"
                               "void main()\n"
                               "{\n"
                               "    TexCoords = vertex.zw;\n"
                               "    gl_Position = projection * view * model * vec4(vertex.xy, 0.0, 1.0);\n"
                               "}\n";

    std::string fragmentShader = "#version 430 core\n"
                                 "in vec2 TexCoords;\n"
                                 "out vec4 color;\n"
                                 "uniform sampler2D image;\n"
                                 "uniform vec4 spriteColor;\n"
                                 "void main()\n"
                                 "{\n"
                                 "  color = spriteColor * texture(image, TexCoords);\n"
                                 "}\n";

    unsigned int program = 0;
    if (Graphics::compile("Sprite", vertexShader, fragmentShader, "", &program))
    {
        state.mSpriteShaderProgram = program;
        state.mSpriteModelLoc = Graphics::findUniformLocation("model", state.mSpriteShaderProgram);
        state.mSpriteViewLoc = Graphics::findUniformLocation("view", state.mSpriteShaderProgram);
        state.mSpriteProjectionLoc = Graphics::findUniformLocation("projection", state.mSpriteShaderProgram);
        state.mSpriteColorLoc = Graphics::findUniformLocation("spriteColor", state.mSpriteShaderProgram);
        state.mSpriteImageLoc = Graphics::findUniformLocation("image", state.mSpriteShaderProgram);
    }
    else
    {
        state.mSpriteShaderProgram = -1;
    }
}

void Graphics::compileGBufferShader(DeferredRendererState &state)
{
    std::string vertexShader = "#version 430 core\n"
                               "layout(location = 0) in vec3 aPos;\n"
                               "layout(location = 1) in vec3 aNormal;\n"
                               "layout(location = 2) in vec2 aTexCoords;\n"
                               "layout(std140) uniform CameraBlock\n"
                               "{\n"
                               "    mat4 projection;\n"
                               "    mat4 view;\n"
                               "    vec3 cameraPos;\n"
                               "}Camera;\n"
                               "out vec3 FragPos;\n"
                               "out vec2 TexCoords;\n"
                               "out vec3 Normal;\n"
                               "uniform mat4 model;\n"
                               "void main()\n"
                               "{\n"
                               "    vec4 worldPos = model * vec4(aPos, 1.0);\n"
                               "    FragPos = worldPos.xyz;\n"
                               "    TexCoords = aTexCoords;\n"
                               "    mat3 normalMatrix = transpose(inverse(mat3(model)));\n"
                               "    Normal = normalMatrix * aNormal;\n"
                               "    gl_Position = Camera.projection * Camera.view * worldPos;\n"
                               "}\n";

    std::string fragmentShader = "#version 430 core\n"
                                 "layout(location = 0) out vec3 gPosition;\n"
                                 "layout(location = 1) out vec3 gNormal;\n"
                                 "layout(location = 2) out vec4 gAlbedoSpec;\n"
                                 "in vec2 TexCoords;\n"
                                 "in vec3 FragPos;\n"
                                 "in vec3 Normal;\n"
                                 "uniform sampler2D texture_diffuse1;\n"
                                 "uniform sampler2D texture_specular1;\n"
                                 "void main()\n"
                                 "{\n"
                                 "  // store the fragment position vector in the first gbuffer texture\n"
                                 "  gPosition = FragPos;\n"
                                 "  // also store the per-fragment normals into the gbuffer\n"
                                 "  gNormal = normalize(Normal);\n"
                                 "  // and the diffuse per-fragment color\n"
                                 "  gAlbedoSpec.rgb = texture(texture_diffuse1, TexCoords).rgb;\n"
                                 "  // store specular intensity in gAlbedoSpec's alpha component\n"
                                 "  gAlbedoSpec.a = texture(texture_specular1, TexCoords).r;\n"
                                 "}\n";

    unsigned int program = 0;
    if (Graphics::compile("GBuffer", vertexShader, fragmentShader, "", &program))
    {
        state.mGBufferShaderProgram = program;
        state.mGBufferShaderModelLoc = Graphics::findUniformLocation("model", state.mGBufferShaderProgram);
        state.mGBufferShaderDiffuseTexLoc = Graphics::findUniformLocation("texture_diffuse1", state.mGBufferShaderProgram);
        state.mGBufferShaderSpecTexLoc = Graphics::findUniformLocation("texture_specular1", state.mGBufferShaderProgram);

        Graphics::setUniformBlock("CameraBlock", 0, state.mGBufferShaderProgram);
    }
    else
    {
        state.mGBufferShaderProgram = -1;
    }
}

void Graphics::compileScreenQuadShader(DeferredRendererState &state)
{
    std::string vertexShader = "#version 330 core\n"
                               "layout(location = 0) in vec2 aPos;\n"
                               "layout(location = 1) in vec2 aTexCoords;\n"
                               "out vec2 TexCoords;\n"
                               "void main()\n"
                               "{\n"
                               "    TexCoords = aTexCoords;\n"
                               "    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);\n"
                               "}\n";

    std::string fragmentShader = "#version 330 core\n"
                                 "out vec4 FragColor;\n"
                                 "in vec2 TexCoords;\n"
                                 "uniform sampler2D screenTexture;\n"
                                 "void main()\n"
                                 "{\n"
                                 "  vec3 col = texture(screenTexture, TexCoords).rgb;\n"
                                 "  FragColor = vec4(col, 1.0);\n"
                                 "}\n";

    unsigned int program = 0;
    if (Graphics::compile("Screen Quad", vertexShader, fragmentShader, "", &program))
    {
        state.mQuadShaderProgram = program;
        state.mQuadShaderTexLoc = Graphics::findUniformLocation("screenTexture", state.mQuadShaderProgram);
    }
    else
    {
        state.mQuadShaderProgram = -1;
    }
}

void Graphics::compileColorShader(DeferredRendererState &state)
{
    std::string vertexShader = "#version 430 core\n"
                               "layout(std140) uniform CameraBlock\n"
                               "{\n"
                               "    mat4 projection;\n"
                               "    mat4 view;\n"
                               "    vec3 cameraPos;\n"
                               "}Camera;\n"
                               "uniform mat4 model;\n"
                               "in vec3 position;\n"
                               "void main()\n"
                               "{\n"
                               "    gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);\n"
                               "}\n";

    std::string fragmentShader = "#version 430 core\n"
                                 "struct Material\n"
                                 "{\n"
                                 "    uvec4 color;\n"
                                 "};\n"
                                 "uniform Material material;\n"
                                 "out vec4 FragColor;\n"
                                 "void main()\n"
                                 "{\n"
                                 "    FragColor = vec4(material.color.r / 255.0f, material.color.g / 255.0f,\n"
                                 "                      material.color.b / 255.0f, material.color.a / 255.0f);\n"
                                 "}\n";

    unsigned int program = 0;
    if (Graphics::compile("Color", vertexShader, fragmentShader, "", &program))
    {
        state.mColorShaderProgram = program;
        state.mColorShaderModelLoc = Graphics::findUniformLocation("model", state.mColorShaderProgram);
        state.mColorShaderColorLoc = Graphics::findUniformLocation("material.color", state.mColorShaderProgram);

        Graphics::setUniformBlock("CameraBlock", 0, state.mColorShaderProgram);
    }
    else
    {
        state.mColorShaderProgram = -1;
    }
}

void Graphics::compileLineShader(GizmoRendererState &state)
{
    std::string vertexShader = "#version 430 core\n"
                               "layout(location = 0) in vec3 position;\n"
                               "layout(location = 1) in vec4 color;\n"
                               "uniform mat4 mvp;\n"
                               "out vec4 Color;\n"
                               "void main()\n"
                               "{\n"
                               "    Color = color;\n"
                               "    gl_Position = mvp * vec4(position, 1.0);\n"
                               "}\n";

    std::string fragmentShader = "#version 430 core\n"
                                 "in vec4 Color;\n"
                                 "out vec4 FragColor;\n"
                                 "void main()\n"
                                 "{\n"
                                 "  FragColor = Color;\n"
                                 "}\n";

    unsigned int program = 0;
    if (Graphics::compile("Line", vertexShader, fragmentShader, "", &program))
    {
        state.mLineShaderProgram = program;
        state.mLineShaderMVPLoc = Graphics::findUniformLocation("mvp", state.mLineShaderProgram);
    }
    else
    {
        state.mLineShaderProgram = -1;
    }
}

void Graphics::compileGizmoShader(GizmoRendererState &state)
{
    std::string vertexShader = "#version 430 core\n"
                               "layout(location = 0) in vec3 position;\n"
                               "layout(location = 1) in vec3 normal;\n"
                               "out vec3 FragPos;\n"
                               "out vec3 Normal;\n"
                               "uniform mat4 model;\n"
                               "uniform mat4 view;\n"
                               "uniform mat4 projection;\n"
                               "void main()\n"
                               "{\n"
                               "    FragPos = vec3(model * vec4(position, 1.0));\n"
                               "    Normal = mat3(transpose(inverse(model))) * normal;\n"
                               "    gl_Position = projection * view * vec4(FragPos, 1.0);\n"
                               "}\n";


    std::string fragmentShader = "#version 430 core\n"
                                 "out vec4 FragColor;\n"
                                 "in vec3 Normal;\n"
                                 "in vec3 FragPos;\n"
                                 "uniform vec3 lightPos;\n"
                                 "uniform vec4 color;\n"
                                 "void main()\n"
                                 "{\n"
                                 "  vec3 norm = normalize(Normal);\n"
                                 "  vec3 lightDir = normalize(lightPos - FragPos);\n"
                                 "  float diff = max(abs(dot(norm, lightDir)), 0.1);\n"
                                 "  vec4 diffuse = vec4(diff, diff, diff, 1.0);\n"
                                 "  FragColor = diffuse * color;\n"
                                 "}\n";

    unsigned int program = 0;
    if (Graphics::compile("Gizmo", vertexShader, fragmentShader, "", &program))
    {
        state.mGizmoShaderProgram = program;
        state.mGizmoShaderColorLoc = Graphics::findUniformLocation("color", state.mGizmoShaderProgram);
        state.mGizmoShaderLightPosLoc = Graphics::findUniformLocation("lightPos", state.mGizmoShaderProgram);
        state.mGizmoShaderModelLoc = Graphics::findUniformLocation("model", state.mGizmoShaderProgram);
        state.mGizmoShaderViewLoc = Graphics::findUniformLocation("view", state.mGizmoShaderProgram);
        state.mGizmoShaderProjLoc = Graphics::findUniformLocation("projection", state.mGizmoShaderProgram);
    }
    else
    {
        state.mGizmoShaderProgram = -1;
    }
}

void Graphics::compileGridShader(GizmoRendererState &state)
{
    std::string vertexShader = "#version 430 core\n"
                               "layout(std140) uniform CameraBlock\n"
                               "{\n"
                               "    mat4 projection;\n"
                               "    mat4 view;\n"
                               "    vec3 cameraPos;\n"
                               "}Camera;\n"
                               "uniform mat4 mvp;\n"
                               "uniform vec4 color;\n"
                               "in vec3 position;\n"
                               "out vec4 Color;\n"
                               "void main()\n"
                               "{\n"
                               "    gl_Position = mvp * vec4(position, 1.0);\n"
                               "    Color = color;\n"
                               "}\n";

    std::string fragmentShader = "#version 430 core\n"
                                 "in vec4 Color;\n"
                                 "out vec4 FragColor;\n"
                                 "void main()\n"
                                 "{\n"
                                 "  float depth = 0.2f * gl_FragCoord.z / gl_FragCoord.w;\n"
                                 "  FragColor = vec4(Color.x, Color.y, Color.z, clamp(1.0f / depth, 0.0f, 0.8f));\n"
                                 "}\n";

    unsigned int program = 0;
    if (Graphics::compile("Grid", vertexShader, fragmentShader, "", &program))
    {
        state.mGridShaderProgram = program;
        state.mGridShaderMVPLoc = Graphics::findUniformLocation("mvp", state.mGridShaderProgram);
        state.mGridShaderColorLoc = Graphics::findUniformLocation("color", state.mGridShaderProgram);

        Graphics::setUniformBlock("CameraBlock", 0, state.mGridShaderProgram);
    }
    else
    {
        state.mGridShaderProgram = -1;
    }
}

void Graphics::createFrustum(const std::vector<float> &vertices, const std::vector<float> &normals, unsigned int *vao,
                          unsigned int *vbo0, unsigned int *vbo1)
{
    glGenVertexArrays(1, vao);
    glBindVertexArray(*vao);

    glGenBuffers(2, vbo0);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo0);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    glGenBuffers(1, vbo1);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo1);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), &normals[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Graphics::destroyFrustum(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1)
{
    glDeleteVertexArrays(1, vao);
    glDeleteBuffers(2, vbo0);
    glDeleteBuffers(2, vbo1);
}

void Graphics::createGrid(const std::vector<glm::vec3> &vertices, unsigned int *vao, unsigned int *vbo0)
{
    glGenVertexArrays(1, vao);
    glBindVertexArray(*vao);

    glGenBuffers(1, vbo0);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo0);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Graphics::destroyGrid(unsigned int *vao, unsigned int *vbo0)
{
    glDeleteVertexArrays(1, vao);
    glDeleteBuffers(1, vbo0);
}

void Graphics::createLine(const std::vector<float> &vertices, const std::vector<float> &colors, unsigned int *vao,
                          unsigned int *vbo0, unsigned int *vbo1)
{
    glGenVertexArrays(1, vao);
    glBindVertexArray(*vao);

    glGenBuffers(2, vbo0);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo0);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    glGenBuffers(1, vbo1);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo1);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float), &colors[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GL_FLOAT), 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Graphics::destroyLine(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1)
{
    glDeleteVertexArrays(1, vao);
    glDeleteBuffers(1, vbo0);
    glDeleteBuffers(1, vbo1);
}