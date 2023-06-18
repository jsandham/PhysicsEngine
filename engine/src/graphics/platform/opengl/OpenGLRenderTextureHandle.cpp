#include "../../../../include/graphics/platform/opengl/OpenGLRenderTextureHandle.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"
#include "../../../../include/core/Log.h"

#include <GL/glew.h>
#include <glm/glm.hpp>

using namespace PhysicsEngine;

static GLenum getTextureFormat(TextureFormat format)
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

static GLint getTextureWrapMode(TextureWrapMode wrapMode)
{
    GLint openglWrapMode = GL_REPEAT;

    switch (wrapMode)
    {
    case TextureWrapMode::Repeat:
        openglWrapMode = GL_REPEAT;
        break;
    case TextureWrapMode::ClampToEdge:
        openglWrapMode = GL_CLAMP_TO_EDGE;
        break;
    case TextureWrapMode::ClampToBorder:
        openglWrapMode = GL_CLAMP_TO_BORDER;
        break;
    case TextureWrapMode::MirrorRepeat:
        openglWrapMode = GL_MIRRORED_REPEAT;
        break;
    case TextureWrapMode::MirrorClampToEdge:
        openglWrapMode = GL_MIRROR_CLAMP_TO_EDGE;
        break;
    default:
        Log::error("OpengGL: Invalid texture wrap mode\n");
        break;
    }

    return openglWrapMode;
}

static GLint getTextureFilterMode(TextureFilterMode filterMode)
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


OpenGLRenderTextureHandle::OpenGLRenderTextureHandle(int width, int height, TextureFormat format,
                                                     TextureWrapMode wrapMode, TextureFilterMode filterMode)
    : RenderTextureHandle(width, height, format, wrapMode, filterMode)
{
    CHECK_ERROR(glGenTextures(1, &mHandle));

    mFormat = format;
    mWrapMode = wrapMode;
    mFilterMode = filterMode;
    mAnisoLevel = 1;
    mWidth = width;
    mHeight = height;

    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, mHandle));

    GLenum openglFormat = getTextureFormat(mFormat);
    GLint openglFilterMode = getTextureFilterMode(mFilterMode);

    // glTexImage2D allows "re-allocating" a texture without having to delete and re-create it first
    if (mFormat == TextureFormat::Depth)
    {
        CHECK_ERROR(
            glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, mWidth, mHeight, 0, openglFormat, GL_FLOAT, nullptr));
    }
    else
    {
        CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, mWidth, mHeight, 0, openglFormat, GL_UNSIGNED_BYTE,
                                 nullptr));
    }

    CHECK_ERROR(glGenerateMipmap(GL_TEXTURE_2D));

    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, openglFilterMode));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, getTextureWrapMode(mWrapMode)));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, getTextureWrapMode(mWrapMode)));

    // clamp the requested anisotropic filtering level to what is available and set it
    CHECK_ERROR(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 1.0f));

    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
}

OpenGLRenderTextureHandle::~OpenGLRenderTextureHandle()
{
    CHECK_ERROR(glDeleteTextures(1, &mHandle));
}

void *OpenGLRenderTextureHandle::getTexture()
{
    return static_cast<void*>(&mHandle);
}

void *OpenGLRenderTextureHandle::getIMGUITexture()
{
    return static_cast<void*>(&mHandle);
}