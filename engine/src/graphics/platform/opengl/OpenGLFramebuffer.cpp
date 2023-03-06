#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"
#include "../../../../include/graphics/platform/opengl/OpenGLFramebuffer.h"

#include <GL/glew.h>
#include <glm/glm.hpp>

using namespace PhysicsEngine;

OpenGLFramebuffer::OpenGLFramebuffer(int width, int height) : Framebuffer(width, height)
{
    mColorTex.resize(1);

    CHECK_ERROR(glGenFramebuffers(1, &mHandle));
    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, mHandle));

    mColorTex[0] = TextureHandle::create(mWidth, mHeight, TextureFormat::RGBA, TextureWrapMode::ClampToEdge,
                                          TextureFilterMode::Nearest);
    mDepthTex = TextureHandle::create(mWidth, mHeight, TextureFormat::Depth, TextureWrapMode::ClampToEdge,
                                          TextureFilterMode::Nearest);   

    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                           *reinterpret_cast<unsigned int *>(mColorTex[0]->getHandle()), 0));
    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                                           *reinterpret_cast<unsigned int *>(mDepthTex->getHandle()), 0));

    // - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
    unsigned int mainAttachments[1] = {GL_COLOR_ATTACHMENT0};
    CHECK_ERROR(glDrawBuffers(1, mainAttachments));

    checkFrambufferError(std::to_string(__LINE__), std::string(__FILE__));

    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

OpenGLFramebuffer::OpenGLFramebuffer(int width, int height, int numColorTex, bool addDepthTex) : Framebuffer(width, height, numColorTex, addDepthTex)
{
    mColorTex.resize(mNumColorTex);

    CHECK_ERROR(glGenFramebuffers(1, &mHandle));
    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, mHandle));

    for (size_t i = 0; i < mColorTex.size(); i++)
    {
        mColorTex[i] = TextureHandle::create(mWidth, mHeight, TextureFormat::RGBA, TextureWrapMode::ClampToEdge,
                                             TextureFilterMode::Nearest);

        CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, static_cast<GLenum>(GL_COLOR_ATTACHMENT0 + i), GL_TEXTURE_2D,
                                           *reinterpret_cast<unsigned int *>(mColorTex[i]->getHandle()), 0));
    }

    if (mAddDepthTex)
    {
        mDepthTex = TextureHandle::create(mWidth, mHeight, TextureFormat::Depth, TextureWrapMode::ClampToEdge,
                                          TextureFilterMode::Nearest);

        CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                                           *reinterpret_cast<unsigned int *>(mDepthTex->getHandle()), 0));
    }
    else
    {
        mDepthTex = nullptr;
    }

    // - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
    std::vector<unsigned int> mainAttachments(mNumColorTex);
    for (size_t i = 0; i < mainAttachments.size(); i++)
    {
        mainAttachments[i] = GL_COLOR_ATTACHMENT0 + static_cast<unsigned int>(i);
    }
    CHECK_ERROR(glDrawBuffers(mNumColorTex, mainAttachments.data()));

    checkFrambufferError(std::to_string(__LINE__), std::string(__FILE__));

    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

OpenGLFramebuffer::~OpenGLFramebuffer()
{
    // detach textures from their framebuffer
    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, mHandle));
    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0));
    CHECK_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0));
    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    // delete frambuffer
    CHECK_ERROR(glDeleteFramebuffers(1, &mHandle));

    // delete textures
    for (size_t i = 0; i < mColorTex.size(); i++)
    {
        delete mColorTex[i];
    }
    
    if (mAddDepthTex)
    {
        delete mDepthTex;   
    }
}

void OpenGLFramebuffer::clearColor(Color color)
{
    CHECK_ERROR(glClearColor(color.mR, color.mG, color.mB, color.mA));
    CHECK_ERROR(glClear(GL_COLOR_BUFFER_BIT));
}

void OpenGLFramebuffer::clearColor(float r, float g, float b, float a)
{
    CHECK_ERROR(glClearColor(r, g, b, a));
    CHECK_ERROR(glClear(GL_COLOR_BUFFER_BIT));
}

void OpenGLFramebuffer::clearDepth(float depth)
{
    CHECK_ERROR(glClearDepth(depth));
    CHECK_ERROR(glClear(GL_DEPTH_BUFFER_BIT));
}

void OpenGLFramebuffer::bind()
{
    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, mHandle));
}

void OpenGLFramebuffer::unbind()
{
    CHECK_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

void OpenGLFramebuffer::setViewport(int x, int y, int width, int height)
{
    assert(x >= 0);
    assert(y >= 0);
    assert((unsigned int)(x + width) <= mWidth);
    assert((unsigned int)(y + height) <= mHeight);

    CHECK_ERROR(glViewport(x, y, width, height));
}

TextureHandle *OpenGLFramebuffer::getColorTex(size_t i)
{
    assert(i < mColorTex.size());
    return mColorTex[i];
}

TextureHandle *OpenGLFramebuffer::getDepthTex()
{
    return mDepthTex;
}

void *OpenGLFramebuffer::getHandle()
{
    return static_cast<void *>(&mHandle);
}