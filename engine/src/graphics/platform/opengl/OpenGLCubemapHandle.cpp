#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/opengl/OpenGLError.h"
#include "../../../../include/graphics/platform/opengl/OpenGLTextureHandle.h"
#include "../../../../include/graphics/platform/opengl/OpenGLCubemapHandle.h"

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

OpenGLCubemapHandle::OpenGLCubemapHandle(int width, TextureFormat format, TextureWrapMode wrapMode,
                                         TextureFilterMode filterMode)
    : CubemapHandle(width, format, wrapMode, filterMode)
{
    CHECK_ERROR(glGenTextures(1, &mHandle));

    this->load(format, wrapMode, filterMode, width, std::vector<unsigned char>());
}

OpenGLCubemapHandle::~OpenGLCubemapHandle()
{
    CHECK_ERROR(glDeleteTextures(1, &mHandle));
}

void OpenGLCubemapHandle::load(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                               const std::vector<unsigned char> &data)
{
    mFormat = format;
    mWrapMode = wrapMode;
    mFilterMode = filterMode;
    mWidth = width;

    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, mHandle));

    GLenum openglFormat = getTextureFormat(mFormat);
    GLint openglFilterMode = getTextureFilterMode(mFilterMode);

    for (unsigned int i = 0; i < 6; i++)
    {
        CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, mWidth, mWidth, 0, openglFormat,
                                 GL_UNSIGNED_BYTE, data.data()));
    }

    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, openglFilterMode));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER,
                                openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, getTextureWrapMode(mWrapMode)));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, getTextureWrapMode(mWrapMode)));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, getTextureWrapMode(mWrapMode)));

    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, 0));
}

void OpenGLCubemapHandle::update(TextureWrapMode wrapMode, TextureFilterMode filterMode)
{
    mWrapMode = wrapMode;
    mFilterMode = filterMode;

    GLint openglFilterMode = getTextureFilterMode(mFilterMode);
    
    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, mHandle));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, openglFilterMode));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER,
                    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, getTextureWrapMode(mWrapMode)));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, getTextureWrapMode(mWrapMode)));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, getTextureWrapMode(mWrapMode)));
    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, 0));
}

void OpenGLCubemapHandle::readPixels(std::vector<unsigned char> &data)
{
    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, mHandle));
    
    GLenum openglFormat = getTextureFormat(mFormat);
    
    int numChannels = 1;
    switch (mFormat)
    {
    case TextureFormat::Depth:
        numChannels = 1;
        break;
    case TextureFormat::RG:
        numChannels = 2;
        break;
    case TextureFormat::RGB:
        numChannels = 3;
        break;
    case TextureFormat::RGBA:
        numChannels = 4;
        break;
    }

    for (unsigned int i = 0; i < 6; i++)
    {
        CHECK_ERROR(glGetTexImage(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, GL_UNSIGNED_BYTE,
                        &data[i * mWidth * mWidth * numChannels]));
    }
    
    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, 0));
}

void OpenGLCubemapHandle::writePixels(const std::vector<unsigned char> &data)
{
    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, mHandle));
    
    GLenum openglFormat = getTextureFormat(mFormat);
    
    for (unsigned int i = 0; i < 6; i++)
    {
        CHECK_ERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, mWidth, mWidth, 0, openglFormat,
                        GL_UNSIGNED_BYTE, data.data()));
    }
    
    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, 0));
}

void OpenGLCubemapHandle::bind(unsigned int texUnit)
{
    CHECK_ERROR(glActiveTexture(GL_TEXTURE0 + texUnit));
    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, mHandle));
}

void OpenGLCubemapHandle::unbind(unsigned int texUnit)
{
    CHECK_ERROR(glActiveTexture(GL_TEXTURE0 + texUnit));
    CHECK_ERROR(glBindTexture(GL_TEXTURE_CUBE_MAP, 0));
}

void *OpenGLCubemapHandle::getHandle()
{
    return static_cast<void *>(&mHandle);
}