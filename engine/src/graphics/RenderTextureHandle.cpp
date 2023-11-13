#include "../../include/graphics/RenderTextureHandle.h"
#include "../../include/graphics/RenderContext.h"

#include "../../include/graphics/platform/directx/DirectXRenderTextureHandle.h"
#include "../../include/graphics/platform/opengl/OpenGLRenderTextureHandle.h"

using namespace PhysicsEngine;

RenderTextureHandle::RenderTextureHandle(int width, int height, TextureFormat format, TextureWrapMode wrapMode,
                                         TextureFilterMode filterMode)
{
    mWidth = width;
    mHeight = height;
    mAnisoLevel = 1;

    mFormat = format;
    mWrapMode = wrapMode;
    mFilterMode = filterMode;
}

RenderTextureHandle::~RenderTextureHandle()
{
}

RenderTextureHandle *RenderTextureHandle::create(int width, int height, TextureFormat format, TextureWrapMode wrapMode,
                                                 TextureFilterMode filterMode)
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLRenderTextureHandle(width, height, format, wrapMode, filterMode);
    case RenderAPI::DirectX:
        return new DirectXRenderTextureHandle(width, height, format, wrapMode, filterMode);
    }

    return nullptr;
}

int RenderTextureHandle::getWidth() const
{
    return mWidth;
}

int RenderTextureHandle::getHeight() const
{
    return mHeight;
}