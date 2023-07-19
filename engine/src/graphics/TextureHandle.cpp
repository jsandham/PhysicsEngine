#include "../../include/graphics/TextureHandle.h"
#include "../../include/graphics/RenderContext.h"

#include "../../include/graphics/platform/directx/DirectXTextureHandle.h"
#include "../../include/graphics/platform/opengl/OpenGLTextureHandle.h"

using namespace PhysicsEngine;

TextureHandle::TextureHandle(int width, int height, TextureFormat format, TextureWrapMode wrapMode,
                             TextureFilterMode filterMode)
{
    mWidth = width;
    mHeight = height;
    mAnisoLevel = 1;

    mFormat = format;
    mWrapMode = wrapMode;
    mFilterMode = filterMode;
}

TextureHandle::~TextureHandle()
{
}

TextureFormat TextureHandle::getFormat() const
{
    return mFormat;
}

TextureWrapMode TextureHandle::getWrapMode() const
{
    return mWrapMode;
}

TextureFilterMode TextureHandle::getFilterMode() const
{
    return mFilterMode;
}

int TextureHandle::getAnisoLevel() const
{
    return mAnisoLevel;
}

int TextureHandle::getWidth() const
{
    return mWidth;
}

int TextureHandle::getHeight() const
{
    return mHeight;
}

TextureHandle *TextureHandle::create(int width, int height, TextureFormat format, TextureWrapMode wrapMode,
                                     TextureFilterMode filterMode)
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLTextureHandle(width, height, format, wrapMode, filterMode);
    case RenderAPI::DirectX:
        return new DirectXTextureHandle(width, height, format, wrapMode, filterMode);
    }

    return nullptr;
}
