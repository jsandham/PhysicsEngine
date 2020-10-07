#include <GL/glew.h>
#include <algorithm>
#include <iostream>
#include <random>

#include "../../include/core/InternalShaders.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"
#include "../../include/graphics/GraphicsState.h"

using namespace PhysicsEngine;

void Graphics::checkError()
{
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR)
    {
        std::string errorStr;
        switch (error)
        {
        case GL_INVALID_ENUM:
            errorStr = "Error: An unacceptable value is specified for an enumerated argument";
            break;
        case GL_INVALID_VALUE:
            errorStr = "Error: A numeric argument is out of range";
            break;
        case GL_INVALID_OPERATION:
            errorStr = "Error: The specified operation is not allowed in the current state";
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            errorStr = "Error: The framebuffer object is not complete";
            break;
        case GL_OUT_OF_MEMORY:
            errorStr = "Error: There is not enough money left to execute the command";
            break;
        case GL_STACK_UNDERFLOW:
            errorStr = "Error: An attempt has been made to perform an operation that would cause an internal stack to "
                       "underflow";
            break;
        case GL_STACK_OVERFLOW:
            errorStr = "Error: An attempt has been made to perform an operation that would cause an internal stack to "
                       "overflow";
            break;
        default:
            errorStr = "Error: Unknown error";
            break;
        }

        std::string errorMessage = errorStr + "(" + std::to_string(error) + ")\n";
        Log::error(errorMessage.c_str());
    }
}

void Graphics::checkFrambufferError()
{
    GLenum framebufferStatus = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
    if (framebufferStatus != GL_FRAMEBUFFER_COMPLETE)
    {
        std::string errorStr;
        switch (framebufferStatus)
        {
        case GL_FRAMEBUFFER_UNDEFINED:
            errorStr = "Error: The current FBO binding is 0 but no default framebuffer exists";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            errorStr = "Error: One of the buffers enabled for rendering is incomplete";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            errorStr =
                "Error: No buffers are attached to the FBO and it is not configured for rendering without attachments";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
            errorStr = "Error: Not all attachments enabled via glDrawBuffers exists in framebuffer";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
            errorStr = "Error: Not all buffers specified via glReadBuffer exists in framebuffer";
            break;
        case GL_FRAMEBUFFER_UNSUPPORTED:
            errorStr = "Error: The combination of internal buffer formats is unsupported";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
            errorStr = "Error: The number of samples for each attachment is not the same";
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
            errorStr = "Error: Not all color attachments are layered textures or bound to the same target";
            break;
        default:
            errorStr = "Error: Unknown framebuffer status error";
            break;
        }

        std::string errorMessage = errorStr + "(" + std::to_string(framebufferStatus) + ")\n";
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
    }

    return openglFormat;
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

void Graphics::createTargets(CameraTargets *targets, Viewport viewport, glm::vec3 *ssaoSamples, GLuint *queryId0,
                             GLuint *queryId1, bool *created)
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

    Graphics::checkFrambufferError();

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

    Graphics::checkFrambufferError();

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

    Graphics::checkFrambufferError();

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

    Graphics::checkFrambufferError();

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

    Graphics::checkError();

    *created = true;
}

void Graphics::destroyTargets(CameraTargets *targets, GLuint *queryId0, GLuint *queryId1, bool *created)
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

    *created = false;
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
    Graphics::checkError();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Graphics::createTargets(LightTargets *targets, ShadowMapResolution resolution, bool *created)
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

        Graphics::checkFrambufferError();

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

    Graphics::checkFrambufferError();

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

    Graphics::checkFrambufferError();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    Graphics::checkError();

    *created = true;
}

void Graphics::destroyTargets(LightTargets *targets, bool *created)
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

    Graphics::checkError();

    *created = false;
}

void Graphics::resizeTargets(LightTargets *targets, ShadowMapResolution resolution, bool *resolutionChanged)
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

    Graphics::checkError();

    *resolutionChanged = false;
}

void Graphics::create(Texture2D *texture, GLuint *tex, bool *created)
{
    int width = texture->getWidth();
    int height = texture->getHeight();
    TextureFormat format = texture->getFormat();
    std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_2D, *tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);

    glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindTexture(GL_TEXTURE_2D, 0);

    *created = true;

    Graphics::checkError();
}

void Graphics::destroy(Texture2D *texture, GLuint *tex, bool *created)
{
    glDeleteTextures(1, tex);

    *created = false;
}

void Graphics::readPixels(Texture2D *texture)
{
    int width = texture->getWidth();
    int height = texture->getHeight();
    int numChannels = texture->getNumChannels();
    TextureFormat format = texture->getFormat();
    std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

    glBindTexture(GL_TEXTURE_2D, texture->getNativeGraphics());

    GLenum openglFormat = Graphics::getTextureFormat(format);

    glGetTextureImage(texture->getNativeGraphics(), 0, openglFormat, GL_UNSIGNED_BYTE, width * height * numChannels,
                      &rawTextureData[0]);

    texture->setRawTextureData(rawTextureData, width, height, format);

    glBindTexture(GL_TEXTURE_2D, 0);

    Graphics::checkError();
}

void Graphics::apply(Texture2D *texture)
{
    int width = texture->getWidth();
    int height = texture->getHeight();
    TextureFormat format = texture->getFormat();
    std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

    glBindTexture(GL_TEXTURE_2D, texture->getNativeGraphics());

    GLenum openglFormat = Graphics::getTextureFormat(format);

    glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

    glBindTexture(GL_TEXTURE_2D, 0);

    Graphics::checkError();
}

void Graphics::create(Texture3D *texture, GLuint *tex, bool *created)
{
    int width = texture->getWidth();
    int height = texture->getHeight();
    int depth = texture->getDepth();
    TextureFormat format = texture->getFormat();
    std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_3D, *tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);

    glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE,
                 &rawTextureData[0]);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

    glBindTexture(GL_TEXTURE_3D, 0);

    *created = true;

    Graphics::checkError();
}

void Graphics::destroy(Texture3D *texture, GLuint *tex, bool *created)
{
    glDeleteTextures(1, tex);

    *created = false;
}

void Graphics::readPixels(Texture3D *texture)
{
    int width = texture->getWidth();
    int height = texture->getHeight();
    int depth = texture->getDepth();
    int numChannels = texture->getNumChannels();
    TextureFormat format = texture->getFormat();
    std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

    glBindTexture(GL_TEXTURE_3D, texture->getNativeGraphics());

    GLenum openglFormat = Graphics::getTextureFormat(format);

    glGetTextureImage(texture->getNativeGraphics(), 0, openglFormat, GL_UNSIGNED_BYTE,
                      width * height * depth * numChannels, &rawTextureData[0]);

    texture->setRawTextureData(rawTextureData, width, height, depth, format);

    glBindTexture(GL_TEXTURE_3D, 0);

    Graphics::checkError();
}

void Graphics::apply(Texture3D *texture)
{
    int width = texture->getWidth();
    int height = texture->getHeight();
    int depth = texture->getDepth();
    TextureFormat format = texture->getFormat();
    std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

    glBindTexture(GL_TEXTURE_3D, texture->getNativeGraphics());

    GLenum openglFormat = Graphics::getTextureFormat(format);

    glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE,
                 &rawTextureData[0]);

    glBindTexture(GL_TEXTURE_3D, 0);

    Graphics::checkError();
}

void Graphics::create(Cubemap *cubemap, GLuint *tex, bool *created)
{
    int width = cubemap->getWidth();
    TextureFormat format = cubemap->getFormat();
    std::vector<unsigned char> rawCubemapData = cubemap->getRawCubemapData();

    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, *tex);

    GLenum openglFormat = Graphics::getTextureFormat(format);

    for (unsigned int i = 0; i < 6; i++)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, width, width, 0, openglFormat,
                     GL_UNSIGNED_BYTE, &rawCubemapData[0]);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    *created = true;

    Graphics::checkError();
}

void Graphics::destroy(Cubemap *cubemap, GLuint *tex, bool *created)
{
    glDeleteTextures(1, tex);

    *created = false;
}

void Graphics::readPixels(Cubemap *cubemap)
{
    int width = cubemap->getWidth();
    int numChannels = cubemap->getNumChannels();
    TextureFormat format = cubemap->getFormat();
    std::vector<unsigned char> rawCubemapData = cubemap->getRawCubemapData();

    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap->getNativeGraphics());

    GLenum openglFormat = Graphics::getTextureFormat(format);

    for (unsigned int i = 0; i < 6; i++)
    {
        glGetTexImage(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, GL_UNSIGNED_BYTE,
                      &rawCubemapData[i * width * width * numChannels]);
    }

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    Graphics::checkError();
}

void Graphics::apply(Cubemap *cubemap)
{
    int width = cubemap->getWidth();
    TextureFormat format = cubemap->getFormat();
    std::vector<unsigned char> rawCubemapData = cubemap->getRawCubemapData();

    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap->getNativeGraphics());

    GLenum openglFormat = Graphics::getTextureFormat(format);

    for (unsigned int i = 0; i < 6; i++)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, width, width, 0, openglFormat,
                     GL_UNSIGNED_BYTE, &rawCubemapData[0]);
    }

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    Graphics::checkError();
}

void Graphics::create(Mesh *mesh, GLuint *vao, GLuint *vbo0, GLuint *vbo1, GLuint *vbo2, bool *created)
{
    glGenVertexArrays(1, vao);
    glBindVertexArray(*vao);
    glGenBuffers(1, vbo0);
    glGenBuffers(1, vbo1);
    glGenBuffers(1, vbo2);

    glBindVertexArray(*vao);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo0);
    glBufferData(GL_ARRAY_BUFFER, mesh->getVertices().size() * sizeof(float), &(mesh->getVertices()[0]),
                 GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

    glBindBuffer(GL_ARRAY_BUFFER, *vbo1);
    glBufferData(GL_ARRAY_BUFFER, mesh->getNormals().size() * sizeof(float), &(mesh->getNormals()[0]), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

    glBindBuffer(GL_ARRAY_BUFFER, *vbo2);
    glBufferData(GL_ARRAY_BUFFER, mesh->getTexCoords().size() * sizeof(float), &(mesh->getTexCoords()[0]),
                 GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

    glBindVertexArray(0);

    *created = true;

    Graphics::checkError();
}

void Graphics::destroy(Mesh *mesh, GLuint *vao, GLuint *vbo0, GLuint *vbo1, GLuint *vbo2, bool *created)
{
}

void Graphics::apply(Mesh *mesh)
{
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

void Graphics::render(World *world, RenderObject renderObject, GraphicsQuery *query)
{
    GLsizei numVertices = renderObject.size / 3;

    glBindVertexArray(renderObject.vao);
    glDrawArrays(GL_TRIANGLES, renderObject.start / 3, numVertices);
    glBindVertexArray(0);

    query->mNumDrawCalls++;
    query->mVerts += numVertices;
    query->mTris += numVertices / 3;

    Graphics::checkError();
}