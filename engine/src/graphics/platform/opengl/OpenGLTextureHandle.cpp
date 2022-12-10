#include "../../../../include/graphics/platform/opengl/OpenGLTextureHandle.h"
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
    case TextureWrapMode::Clamp:
        openglWrapMode = GL_CLAMP_TO_EDGE;
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

OpenGLTextureHandle::OpenGLTextureHandle()
{
	CHECK_ERROR(glGenTextures(1, &mHandle));
}

OpenGLTextureHandle::OpenGLTextureHandle(int width, int height, TextureFormat format, TextureWrapMode wrapMode,
                                       TextureFilterMode filterMode)
{
    mFormat = format;
    mWrapMode = wrapMode;
    mFilterMode = filterMode;
    mAnisoLevel = 1;
    mWidth = width;
    mHeight = height;

    CHECK_ERROR(glGenTextures(1, &mHandle));
    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, mHandle));

    GLenum openglFormat = getTextureFormat(mFormat);
    GLint openglWrapMode = getTextureWrapMode(mWrapMode);
    GLint openglFilterMode = getTextureFilterMode(mFilterMode);

    // glTexImage2D allows "re-allocating" a texture without having to delete and re-create it first
    if (mFormat == TextureFormat::Depth)
    {
        CHECK_ERROR(
            glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, mWidth, mHeight, 0, openglFormat, GL_FLOAT, nullptr));
    }
    else
    {
        CHECK_ERROR(
            glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, mWidth, mHeight, 0, openglFormat, GL_UNSIGNED_BYTE, nullptr));
    }

    CHECK_ERROR(glGenerateMipmap(GL_TEXTURE_2D));

    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, openglFilterMode));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, openglWrapMode));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, openglWrapMode));

    // clamp the requested anisotropic filtering level to what is available and set it
    CHECK_ERROR(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 1.0f));

    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
}

OpenGLTextureHandle::~OpenGLTextureHandle()
{
	CHECK_ERROR(glDeleteTextures(1, &mHandle));
}

void OpenGLTextureHandle::load(TextureFormat format,
	TextureWrapMode wrapMode,
	TextureFilterMode filterMode,
	int width,
	int height,
	const std::vector<unsigned char>& data)
{
	mFormat = format;
	mWrapMode = wrapMode;
	mFilterMode = filterMode;
	mAnisoLevel = 1;
	mWidth = width;
	mHeight = height;

	CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, mHandle));

	GLenum openglFormat = getTextureFormat(mFormat);
	GLint openglWrapMode = getTextureWrapMode(mWrapMode);
	GLint openglFilterMode = getTextureFilterMode(mFilterMode);

	// glTexImage2D allows "re-allocating" a texture without having to delete and re-create it first
    if (mFormat == TextureFormat::Depth)
    {
        CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, mWidth, mHeight, 0, openglFormat, GL_FLOAT,
                                 data.data()));
    }
    else
    {
        CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, mWidth, mHeight, 0, openglFormat, GL_UNSIGNED_BYTE,
                                 data.data()));
    }

	CHECK_ERROR(glGenerateMipmap(GL_TEXTURE_2D));

	CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, openglFilterMode));
	CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
		openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode));
	CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, openglWrapMode));
	CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, openglWrapMode));

	// clamp the requested anisotropic filtering level to what is available and set it
	CHECK_ERROR(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 1.0f));

	CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
}

void OpenGLTextureHandle::update(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel)
{
    mWrapMode = wrapMode;
    mFilterMode = filterMode;
    mAnisoLevel = anisoLevel;

    GLint openglWrapMode = getTextureWrapMode(mWrapMode);
    GLint openglFilterMode = getTextureFilterMode(mFilterMode);
    
    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, mHandle));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, openglFilterMode));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    openglFilterMode == GL_LINEAR_MIPMAP_LINEAR ? GL_LINEAR : openglFilterMode));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, openglWrapMode));
    CHECK_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, openglWrapMode));
    
    // Determine how many levels of anisotropic filtering are available
    float aniso = 0.0f;
    CHECK_ERROR(glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso));
    
    // clamp the requested anisotropic filtering level to what is available and set it
    CHECK_ERROR(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, glm::clamp((float)mAnisoLevel, 1.0f,
    aniso)));
    
    CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
}

void OpenGLTextureHandle::readPixels(std::vector<unsigned char>& data)
{
	CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, mHandle));

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

	CHECK_ERROR(glGetTextureImage(mHandle, 0, openglFormat, GL_UNSIGNED_BYTE, mWidth * mHeight * numChannels, &data[0]));

	CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
}

void OpenGLTextureHandle::writePixels(const std::vector<unsigned char>& data)
{
	CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, mHandle));

	GLenum openglFormat = getTextureFormat(mFormat);

	CHECK_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, mWidth, mHeight, 0, openglFormat, GL_UNSIGNED_BYTE, data.data()));

	CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
}

void OpenGLTextureHandle::bind(unsigned int texUnit)
{
	CHECK_ERROR(glActiveTexture(GL_TEXTURE0 + texUnit));
	CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, mHandle));
}

void OpenGLTextureHandle::unbind(unsigned int texUnit)
{
	CHECK_ERROR(glActiveTexture(GL_TEXTURE0 + texUnit));
	CHECK_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
}

void* OpenGLTextureHandle::getHandle()
{
	return static_cast<void*>(&mHandle);
}