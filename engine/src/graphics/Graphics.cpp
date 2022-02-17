#include <GL/glew.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <assert.h>

#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"
#include "GLSL/glsl_shaders.h"

using namespace PhysicsEngine;

int Graphics::INSTANCE_BATCH_SIZE = 1000;

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

GLenum getTextureFormat(TextureFormat format)
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

GLint getTextureWrapMode(TextureWrapMode wrapMode)
{
    GLint openglWrapMode = GL_REPEAT;

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

GLint getTextureFilterMode(TextureFilterMode filterMode)
{
    GLint openglFilterMode = GL_NEAREST;

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

void Graphics::beginQuery(unsigned int queryId)
{
    Graphics::checkError(__LINE__, __FILE__);
    glBeginQuery(GL_TIME_ELAPSED, queryId);
    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::endQuery(unsigned int queryId, unsigned long long *elapsedTime)
{
    Graphics::checkError(__LINE__, __FILE__);
    glEndQuery(GL_TIME_ELAPSED);
    Graphics::checkError(__LINE__, __FILE__);
    glGetQueryObjectui64v(queryId, GL_QUERY_RESULT, elapsedTime);
    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createGlobalCameraUniforms(CameraUniform &uniform)
{
    glGenBuffers(1, &uniform.mBuffer);
    glBindBuffer(GL_UNIFORM_BUFFER, uniform.mBuffer);
    glBufferData(GL_UNIFORM_BUFFER, 204, NULL, GL_DYNAMIC_DRAW);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, uniform.mBuffer, 0, 204);
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
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, uniform.mBuffer, 0, 204);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, 64, glm::value_ptr(uniform.mProjection));
    glBufferSubData(GL_UNIFORM_BUFFER, 64, 64, glm::value_ptr(uniform.mView));
    glBufferSubData(GL_UNIFORM_BUFFER, 128, 64, glm::value_ptr(uniform.mViewProjection));
    glBufferSubData(GL_UNIFORM_BUFFER, 192, 12, glm::value_ptr(uniform.mCameraPos));
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

void Graphics::createFramebuffer(int width, int height, unsigned int *fbo, unsigned int *color)
{
    // generate fbo (color + depth)
    glGenFramebuffers(1, fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, *fbo);

    glGenTextures(1, color);
    glBindTexture(GL_TEXTURE_2D, *color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *color, 0);
    
    // - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
    unsigned int mainAttachments[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, mainAttachments);

    Graphics::checkFrambufferError(__LINE__, __FILE__);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
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
    glClearColor(color.mR, color.mG, color.mB, color.mA);
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

    // dummy query to prevent OpenGL errors from popping out
    // see https://www.lighthouse3d.com/tutorials/opengl-timer-query/
    //glQueryCounter(*queryId1, GL_TIMESTAMP);

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

void Graphics::readColorAtPixel(const unsigned int *fbo, int x, int y, Color32 *color)
{
    glBindFramebuffer(GL_FRAMEBUFFER, *fbo);
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

    GLenum openglFormat = getTextureFormat(format);
    GLint openglWrapMode = getTextureWrapMode(wrapMode);
    GLint openglFilterMode = getTextureFilterMode(filterMode);

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
}

void Graphics::destroyTexture2D(unsigned int *tex)
{
    glDeleteTextures(1, tex);
}

void Graphics::updateTexture2D(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, unsigned int tex)
{
    GLint openglWrapMode = getTextureWrapMode(wrapMode);
    GLint openglFilterMode = getTextureFilterMode(filterMode);

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

    GLenum openglFormat = getTextureFormat(format);

    glGetTextureImage(tex, 0, openglFormat, GL_UNSIGNED_BYTE, width * height * numChannels, &data[0]);

    glBindTexture(GL_TEXTURE_2D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::writePixelsTexture2D(TextureFormat format, int width, int height, const std::vector<unsigned char> &data,
                                    unsigned int tex)
{
    glBindTexture(GL_TEXTURE_2D, tex);

    GLenum openglFormat = getTextureFormat(format);

    glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, data.data());

    glBindTexture(GL_TEXTURE_2D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createTexture3D(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                               int height, int depth, const std::vector<unsigned char> &data, unsigned int *tex)
{
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_3D, *tex);

    GLenum openglFormat = getTextureFormat(format);
    GLint openglWrapMode = getTextureWrapMode(wrapMode);
    GLint openglFilterMode = getTextureFilterMode(filterMode);

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
    GLint openglWrapMode = getTextureWrapMode(wrapMode);
    GLint openglFilterMode = getTextureFilterMode(filterMode);

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

    GLenum openglFormat = getTextureFormat(format);

    glGetTextureImage(tex, 0, openglFormat, GL_UNSIGNED_BYTE, width * height * depth * numChannels, &data[0]);

    glBindTexture(GL_TEXTURE_3D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::writePixelsTexture3D(TextureFormat format, int width, int height, int depth,
                                    const std::vector<unsigned char> &data, unsigned int tex)
{
    glBindTexture(GL_TEXTURE_3D, tex);

    GLenum openglFormat = getTextureFormat(format);

    glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE, &data[0]);

    glBindTexture(GL_TEXTURE_3D, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createCubemap(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                             const std::vector<unsigned char> &data, unsigned int *tex)
{
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, *tex);

    GLenum openglFormat = getTextureFormat(format);
    GLint openglWrapMode = getTextureWrapMode(wrapMode);
    GLint openglFilterMode = getTextureFilterMode(filterMode);

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
    GLint openglWrapMode = getTextureWrapMode(wrapMode);
    GLint openglFilterMode = getTextureFilterMode(filterMode);

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

    GLenum openglFormat = getTextureFormat(format);

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

    GLenum openglFormat = getTextureFormat(format);

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
    
    GLenum openglFormat = getTextureFormat(format);
    //GLint openglWrapMode = getTextureWrapMode(wrapMode);
    //GLint openglFilterMode = getTextureFilterMode(filterMode);
    
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

void Graphics::createTerrainChunk(const std::vector<float> &vertices, const std::vector<float> &normals,
                                  const std::vector<float> &texCoords, int vertexCount,
                                  unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1, unsigned int *vbo2)
{
    glGenVertexArrays(1, vao);
    glBindVertexArray(*vao);
    glGenBuffers(1, vbo0);      // vertex vbo
    glGenBuffers(1, vbo1);      // normals vbo
    glGenBuffers(1, vbo2);      // texcoords vbo

    glBindVertexArray(*vao);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo0);
    glBufferData(GL_ARRAY_BUFFER, 81 * 3 * vertexCount * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

    glBindBuffer(GL_ARRAY_BUFFER, *vbo1);
    glBufferData(GL_ARRAY_BUFFER, 81 * 3 * vertexCount * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, normals.size() * sizeof(float), normals.data());
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

    glBindBuffer(GL_ARRAY_BUFFER, *vbo2);
    glBufferData(GL_ARRAY_BUFFER, 81 * 2 * vertexCount * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, texCoords.size() * sizeof(float), texCoords.data());
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

    glBindVertexArray(0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::destroyTerrainChunk(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1, unsigned int *vbo2)
{
    glDeleteBuffers(1, vbo0);
    glDeleteBuffers(1, vbo1);
    glDeleteBuffers(1, vbo2);

    glDeleteVertexArrays(1, vao);
}

void Graphics::updateTerrainChunk(const std::vector<float> &vertices, const std::vector<float> &normals,
                               unsigned int vbo0, unsigned int vbo1)
{
    glBindBuffer(GL_ARRAY_BUFFER, vbo0);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo1);
    glBufferSubData(GL_ARRAY_BUFFER, 0, normals.size() * sizeof(float), normals.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::updateTerrainChunk(const std::vector<float> &vertices, const std::vector<float> &normals,
                                  const std::vector<float> &texCoords, unsigned int vbo0, unsigned int vbo1,
                                  unsigned int vbo2)
{
    glBindBuffer(GL_ARRAY_BUFFER, vbo0);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo1);
    glBufferSubData(GL_ARRAY_BUFFER, 0, normals.size() * sizeof(float), normals.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo2);
    glBufferSubData(GL_ARRAY_BUFFER, 0, texCoords.size() * sizeof(float), texCoords.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::createMesh(const std::vector<float> &vertices, const std::vector<float> &normals,
                          const std::vector<float> &texCoords, unsigned int*vao, unsigned int*vbo0, unsigned int*vbo1, 
                          unsigned int*vbo2, unsigned int*model_vbo, unsigned int*color_vbo)
{
    glGenVertexArrays(1, vao);
    glBindVertexArray(*vao);
    glGenBuffers(1, vbo0); // vertex vbo
    glGenBuffers(1, vbo1); // normals vbo
    glGenBuffers(1, vbo2); // texcoords vbo
    glGenBuffers(1, model_vbo); // instance model vbo
    glGenBuffers(1, color_vbo); // instance color vbo

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

    // instancing model matrices vbo
    glBindBuffer(GL_ARRAY_BUFFER, *model_vbo);
    glBufferData(GL_ARRAY_BUFFER, INSTANCE_BATCH_SIZE * sizeof(glm::mat4), NULL, GL_DYNAMIC_DRAW);
    // set attribute pointers for matrix (4 times vec4)
    glEnableVertexAttribArray(3); 
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)0);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(sizeof(glm::vec4)));
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(2 * sizeof(glm::vec4)));
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(3 * sizeof(glm::vec4)));

    glVertexAttribDivisor(3, 1);
    glVertexAttribDivisor(4, 1);
    glVertexAttribDivisor(5, 1);
    glVertexAttribDivisor(6, 1);

    // instancing colors vbo
    glBindBuffer(GL_ARRAY_BUFFER, *color_vbo);
    glBufferData(GL_ARRAY_BUFFER, INSTANCE_BATCH_SIZE * sizeof(glm::vec4), NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(7);
    glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void *)0);

    glVertexAttribDivisor(7, 1);

    glBindVertexArray(0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::destroyMesh(unsigned int*vao, unsigned int*vbo0, unsigned int*vbo1, unsigned int*vbo2, unsigned int*model_vbo, unsigned int*color_vbo)
{
    glDeleteBuffers(1, vbo0);
    glDeleteBuffers(1, vbo1);
    glDeleteBuffers(1, vbo2);
    glDeleteBuffers(1, model_vbo);
    glDeleteBuffers(1, color_vbo);

    glDeleteVertexArrays(1, vao);
}

void Graphics::updateInstanceBuffer(unsigned int vbo, const glm::mat4* models, size_t instanceCount)
{
    assert(instanceCount <= INSTANCE_BATCH_SIZE);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, instanceCount * sizeof(glm::mat4), models);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::updateInstanceColorBuffer(unsigned int vbo, const glm::vec4 *colors, size_t instanceCount)
{
    assert(instanceCount <= INSTANCE_BATCH_SIZE);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, instanceCount * sizeof(glm::vec4), colors);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
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

        uniforms[j].mUniformId = 0;
        uniforms[j].mCachedHandle = -1;
        memset(uniforms[j].mData, '\0', 64);
    }

    Graphics::checkError(__LINE__, __FILE__);

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

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::use(int program)
{
    glUseProgram(program);

    Graphics::checkError(__LINE__, __FILE__);
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
    glUniform4fv(nameLocation, 1, static_cast<const GLfloat *>(&color.mR));
}

void Graphics::setColor32(int nameLocation, const Color32& color)
{
    glUniform4ui(nameLocation, static_cast<GLuint>(color.mR), 
        static_cast<GLuint>(color.mG), static_cast<GLuint>(color.mB), static_cast<GLuint>(color.mA));
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
    glGetnUniformfv(program, nameLocation, sizeof(Color), &color.mR);

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

    color.mR = static_cast<unsigned char>(c[0]);
    color.mG = static_cast<unsigned char>(c[1]);
    color.mB = static_cast<unsigned char>(c[2]);
    color.mA = static_cast<unsigned char>(c[3]);

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

void Graphics::applyMaterial(const std::vector<ShaderUniform> &uniforms, int shaderProgram)
{
    Graphics::checkError(__LINE__, __FILE__);
    int textureUnit = 0;
    for (size_t i = 0; i < uniforms.size(); i++)
    {
        int location = findUniformLocation(uniforms[i].mName.c_str(), shaderProgram);

        assert(location != -1);

        if (uniforms[i].mType == ShaderUniformType::Sampler2D)
        {
            if (uniforms[i].mCachedHandle != -1)
            {
                Graphics::setTexture2D(location, textureUnit, uniforms[i].mCachedHandle);
                //Graphics::checkError(__LINE__, __FILE__);
            }
            else
            {
                Graphics::setTexture2D(location, textureUnit, 0);
                //Graphics::checkError(__LINE__, __FILE__);
            }

            textureUnit++;
        }
        else if (uniforms[i].mType == ShaderUniformType::Int)
        {
            Graphics::setInt(location, *reinterpret_cast<const int *>(uniforms[i].mData));
            //Graphics::checkError(__LINE__, __FILE__);
        }
        else if (uniforms[i].mType == ShaderUniformType::Float)
        {
            Graphics::setFloat(location, *reinterpret_cast<const float *>(uniforms[i].mData));
            //Graphics::checkError(__LINE__, __FILE__);
        }
        else if (uniforms[i].mType == ShaderUniformType::Vec2)
        {
            Graphics::setVec2(location, *reinterpret_cast<const glm::vec2 *>(uniforms[i].mData));
            //Graphics::checkError(__LINE__, __FILE__);
        }
        else if (uniforms[i].mType == ShaderUniformType::Vec3)
        {
            Graphics::setVec3(location, *reinterpret_cast<const glm::vec3 *>(uniforms[i].mData));
            //Graphics::checkError(__LINE__, __FILE__);
        }
        else if (uniforms[i].mType == ShaderUniformType::Vec4)
        {
            Graphics::setVec4(location, *reinterpret_cast<const glm::vec4 *>(uniforms[i].mData));
            //Graphics::checkError(__LINE__, __FILE__);
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

void Graphics::renderInstanced(int start, int count, int instanceCount, int vao)
{
    glBindVertexArray(vao);
    glDrawArraysInstanced(GL_TRIANGLES, start, count, instanceCount);
    glBindVertexArray(0);

    Graphics::checkError(__LINE__, __FILE__);
}

void Graphics::render(const RenderObject &renderObject, GraphicsQuery &query)
{
    assert(renderObject.instanced == false);

    int numVertices = renderObject.size / 3;

    Graphics::render(renderObject.start / 3, numVertices, renderObject.vao);

    query.mNumDrawCalls++;
    query.mVerts += numVertices;
    query.mTris += numVertices / 3;
}

void Graphics::renderInstanced(const RenderObject &renderObject, GraphicsQuery &query)
{
    assert(renderObject.instanced == true);

    int numVertices = renderObject.size / 3;

    Graphics::renderInstanced(renderObject.start / 3, numVertices, renderObject.instanceCount,
                              renderObject.vao);

    query.mNumBatchDrawCalls++;
    query.mVerts += numVertices;
    query.mTris += numVertices / 3;
}

void Graphics::compileSSAOShader(ForwardRendererState &state)
{
    unsigned int program = 0;
    if (Graphics::compile("Geometry", getGeometryVertexShader(), getGeometryFragmentShader(), "", &program))
    {
        state.mGeometryShaderProgram = program;
        state.mGeometryShaderModelLoc = Graphics::findUniformLocation("model", state.mGeometryShaderProgram);

        Graphics::setUniformBlock("CameraBlock", 0, state.mGeometryShaderProgram);
    }
    else
    {
        state.mGeometryShaderProgram = -1;
    }

    program = 0;
    if (Graphics::compile("SSAO", getSSAOVertexShader(), getSSAOFragmentShader(), "", &program))
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
    unsigned int program = 0;
    if (Graphics::compile("Shadow Depth Map", getShadowDepthMapVertexShader(), getShadowDepthMapFragmentShader(), "", &program))
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
    unsigned int program = 0;
    if (Graphics::compile("Shadow Depth Cubemap", getShadowDepthCubemapVertexShader(), getShadowDepthCubemapFragmentShader(),
                          getShadowDepthCubemapGeometryShader(),
                          &program))
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
    unsigned int program = 0;
    if (Graphics::compile("Color", getColorVertexShader(), getColorFragmentShader(), "", &program))
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

void Graphics::compileColorInstancedShader(ForwardRendererState &state)
{
    unsigned int program = 0;
    if (Graphics::compile("Color Instanced", getColorInstancedVertexShader(), getColorInstancedFragmentShader(), "", &program))
    {
        state.mColorInstancedShaderProgram = program;

        Graphics::setUniformBlock("CameraBlock", 0, state.mColorInstancedShaderProgram);
    }
    else
    {
        state.mColorInstancedShaderProgram = -1;
    }
}

void Graphics::compileScreenQuadShader(ForwardRendererState &state)
{
    unsigned int program = 0;
    if (Graphics::compile("Screen Quad", getScreenQuadVertexShader(), getScreenQuadFragmentShader(), "", &program))
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
    unsigned int program = 0;
    if (Graphics::compile("Sprite", getSpriteVertexShader(), getSpriteFragmentShader(), "", &program))
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
    unsigned int program = 0;
    if (Graphics::compile("GBuffer", getGBufferVertexShader(), getGBufferFragmentShader(), "", &program))
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
    unsigned int program = 0;
    if (Graphics::compile("Screen Quad", getScreenQuadVertexShader(), getScreenQuadFragmentShader(), "", &program))
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
    unsigned int program = 0;
    if (Graphics::compile("Color", getColorVertexShader(), getColorFragmentShader(), "", &program))
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

void Graphics::compileColorInstancedShader(DeferredRendererState &state)
{
    unsigned int program = 0;
    if (Graphics::compile("Color Instanced", getColorInstancedVertexShader(), getColorInstancedFragmentShader(), "", &program))
    {
        state.mColorInstancedShaderProgram = program;

        Graphics::setUniformBlock("CameraBlock", 0, state.mColorInstancedShaderProgram);
    }
    else
    {
        state.mColorInstancedShaderProgram = -1;
    }
}

void Graphics::compileNormalShader(DebugRendererState &state)
{
    unsigned int program = 0;
    if (Graphics::compile("Normal", getNormalVertexShader(), getNormalFragmentShader(), "", &program))
    {
        state.mNormalsShaderProgram = program;
        state.mNormalsShaderModelLoc = Graphics::findUniformLocation("model", state.mNormalsShaderProgram);

        Graphics::setUniformBlock("CameraBlock", 0, state.mNormalsShaderProgram);
    }
    else
    {
        state.mNormalsShaderProgram = -1;
    }
}

void Graphics::compileNormalInstancedShader(DebugRendererState &state)
{
    unsigned int program = 0;
    if (Graphics::compile("Normal Instanced", getNormalInstancedVertexShader(), getNormalInstancedFragmentShader(), "", &program))
    {
        state.mNormalsInstancedShaderProgram = program;

        Graphics::setUniformBlock("CameraBlock", 0, state.mNormalsInstancedShaderProgram);
    }
    else
    {
        state.mNormalsInstancedShaderProgram = -1;
    }
}


void Graphics::compilePositionShader(DebugRendererState &state)
{
    unsigned int program = 0;
    if (Graphics::compile("Position", getPositionVertexShader(), getPositionFragmentShader(), "", &program))
    {
        state.mPositionShaderProgram = program;
        state.mPositionShaderModelLoc = Graphics::findUniformLocation("model", state.mPositionShaderProgram);

        Graphics::setUniformBlock("CameraBlock", 0, state.mPositionShaderProgram);
    }
    else
    {
        state.mPositionShaderProgram = -1;
    }
}

void Graphics::compilePositionInstancedShader(DebugRendererState &state)
{
    unsigned int program = 0;
    if (Graphics::compile("Position Instanced", getPositionInstancedVertexShader(), getPositionInstancedFragmentShader(), "", &program))
    {
        state.mPositionInstancedShaderProgram = program;

        Graphics::setUniformBlock("CameraBlock", 0, state.mPositionInstancedShaderProgram);
    }
    else
    {
        state.mPositionInstancedShaderProgram = -1;
    }
}

void Graphics::compileLinearDepthShader(DebugRendererState &state)
{
    unsigned int program = 0;
    if (Graphics::compile("Linear Depth", getLinearDepthVertexShader(), getLinearDepthFragmentShader(), "", &program))
    {
        state.mLinearDepthShaderProgram = program;
        state.mLinearDepthShaderModelLoc = Graphics::findUniformLocation("model", state.mLinearDepthShaderProgram);
    
        Graphics::setUniformBlock("CameraBlock", 0, state.mLinearDepthShaderProgram);
    }
    else
    {
        state.mLinearDepthShaderProgram = -1;
    }
}

void Graphics::compileLinearDepthInstancedShader(DebugRendererState &state)
{
    unsigned int program = 0;
    if (Graphics::compile("Linear Depth Instanced", getLinearDepthInstancedVertexShader(), getLinearDepthInstancedFragmentShader(), "", &program))
    {
        state.mLinearDepthInstancedShaderProgram = program;
        Graphics::setUniformBlock("CameraBlock", 0, state.mLinearDepthInstancedShaderProgram);
    }
    else
    {
        state.mLinearDepthInstancedShaderProgram = -1;
    }
}

void Graphics::compileColorShader(DebugRendererState &state)
{
    unsigned int program = 0;
    if (Graphics::compile("Color", getColorVertexShader(), getColorFragmentShader(), "", &program))
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

void Graphics::compileColorInstancedShader(DebugRendererState &state)
{
    unsigned int program = 0;
    if (Graphics::compile("Color Instanced", getColorInstancedVertexShader(), getColorInstancedFragmentShader(), "", &program))
    {
        state.mColorInstancedShaderProgram = program;

        Graphics::setUniformBlock("CameraBlock", 0, state.mColorInstancedShaderProgram);
    }
    else
    {
        state.mColorInstancedShaderProgram = -1;
    }
}

void Graphics::compileScreenQuadShader(DebugRendererState &state)
{
    unsigned int program = 0;
    if (Graphics::compile("Screen Quad", getScreenQuadVertexShader(), getScreenQuadFragmentShader(), "", &program))
    {
        state.mQuadShaderProgram = program;
        state.mQuadShaderTexLoc = Graphics::findUniformLocation("screenTexture", state.mQuadShaderProgram);
    }
    else
    {
        state.mQuadShaderProgram = -1;
    }
}

void Graphics::compileLineShader(GizmoRendererState &state)
{
    unsigned int program = 0;
    if (Graphics::compile("Line", getLineVertexShader(), getLineFragmentShader(), "", &program))
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
    unsigned int program = 0;
    if (Graphics::compile("Gizmo", getGizmoVertexShader(), getGizmoFragmentShader(), "", &program))
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
    unsigned int program = 0;
    if (Graphics::compile("Grid", getGridVertexShader(), getGridFragmentShader(), "", &program))
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

std::string Graphics::getGeometryVertexShader()
{
    return PhysicsEngine::getGeometryVertexShader();
}

std::string Graphics::getGeometryFragmentShader()
{
    return PhysicsEngine::getGeometryFragmentShader();
}

std::string Graphics::getSSAOVertexShader()
{
    return PhysicsEngine::getSSAOVertexShader();
}

std::string Graphics::getSSAOFragmentShader()
{
    return PhysicsEngine::getSSAOFragmentShader();
}

std::string Graphics::getShadowDepthMapVertexShader()
{
    return PhysicsEngine::getShadowDepthMapVertexShader();
}

std::string Graphics::getShadowDepthMapFragmentShader()
{
    return PhysicsEngine::getShadowDepthMapFragmentShader();
}

std::string Graphics::getShadowDepthCubemapVertexShader()
{
    return PhysicsEngine::getShadowDepthCubemapVertexShader();
}

std::string Graphics::getShadowDepthCubemapFragmentShader()
{
    return PhysicsEngine::getShadowDepthCubemapFragmentShader();
}

std::string Graphics::getShadowDepthCubemapGeometryShader()
{
    return PhysicsEngine::getShadowDepthCubemapGeometryShader();
}

std::string Graphics::getColorVertexShader()
{
    return PhysicsEngine::getColorVertexShader();
}

std::string Graphics::getColorFragmentShader()
{
    return PhysicsEngine::getColorFragmentShader();
}

std::string Graphics::getColorInstancedVertexShader()
{
    return PhysicsEngine::getColorInstancedVertexShader();
}

std::string Graphics::getColorInstancedFragmentShader()
{
    return PhysicsEngine::getColorInstancedFragmentShader();
}

std::string Graphics::getScreenQuadVertexShader()
{
    return PhysicsEngine::getScreenQuadVertexShader();
}

std::string Graphics::getScreenQuadFragmentShader()
{
    return PhysicsEngine::getScreenQuadFragmentShader();
}

std::string Graphics::getSpriteVertexShader()
{
    return PhysicsEngine::getSpriteVertexShader();
}

std::string Graphics::getSpriteFragmentShader()
{
    return PhysicsEngine::getSpriteFragmentShader();
}

std::string Graphics::getGBufferVertexShader()
{
    return PhysicsEngine::getGBufferVertexShader();
}

std::string Graphics::getGBufferFragmentShader()
{
    return PhysicsEngine::getGBufferFragmentShader();
}

std::string Graphics::getNormalVertexShader()
{
    return PhysicsEngine::getNormalVertexShader();
}

std::string Graphics::getNormalFragmentShader()
{
    return PhysicsEngine::getNormalFragmentShader();
}

std::string Graphics::getNormalInstancedVertexShader()
{
    return PhysicsEngine::getNormalInstancedVertexShader();
}

std::string Graphics::getNormalInstancedFragmentShader()
{
    return PhysicsEngine::getNormalInstancedFragmentShader();
}

std::string Graphics::getPositionVertexShader()
{
    return PhysicsEngine::getPositionVertexShader();
}

std::string Graphics::getPositionFragmentShader()
{
    return PhysicsEngine::getPositionFragmentShader();
}

std::string Graphics::getPositionInstancedVertexShader()
{
    return PhysicsEngine::getPositionInstancedVertexShader();
}

std::string Graphics::getPositionInstancedFragmentShader()
{
    return PhysicsEngine::getPositionInstancedFragmentShader();
}

std::string Graphics::getLinearDepthVertexShader()
{
    return PhysicsEngine::getLinearDepthVertexShader();
}

std::string Graphics::getLinearDepthFragmentShader()
{
    return PhysicsEngine::getLinearDepthFragmentShader();
}

std::string Graphics::getLinearDepthInstancedVertexShader()
{
    return PhysicsEngine::getLinearDepthInstancedVertexShader();
}

std::string Graphics::getLinearDepthInstancedFragmentShader()
{
    return PhysicsEngine::getLinearDepthInstancedFragmentShader();
}

std::string Graphics::getLineVertexShader()
{
    return PhysicsEngine::getLineVertexShader();
}

std::string Graphics::getLineFragmentShader()
{
    return PhysicsEngine::getLineFragmentShader();
}

std::string Graphics::getGizmoVertexShader()
{
    return PhysicsEngine::getGizmoVertexShader();
}

std::string Graphics::getGizmoFragmentShader()
{
    return PhysicsEngine::getGizmoFragmentShader();
}

std::string Graphics::getGridVertexShader()
{
    return PhysicsEngine::getGridVertexShader();
}

std::string Graphics::getGridFragmentShader()
{
    return PhysicsEngine::getGridFragmentShader();
}

std::string Graphics::getStandardVertexShader()
{
    return PhysicsEngine::getStandardVertexShader();
}

std::string Graphics::getStandardFragmentShader()
{
    return PhysicsEngine::getStandardFragmentShader();
}
