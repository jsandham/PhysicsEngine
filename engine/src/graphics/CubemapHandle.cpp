#include "../../include/graphics/CubemapHandle.h"
#include "../../include/graphics/RenderContext.h"

#include "../../include/graphics/platform/opengl/OpenGLCubemapHandle.h"
#include "../../include/graphics/platform/directx/DirectXCubemapHandle.h"

using namespace PhysicsEngine;

CubemapHandle::CubemapHandle(int width, TextureFormat format, TextureWrapMode wrapMode,
    TextureFilterMode filterMode)
{

}

CubemapHandle::~CubemapHandle()
{
}

TextureFormat CubemapHandle::getFormat() const
{
    return mFormat;
}

TextureWrapMode CubemapHandle::getWrapMode() const
{
    return mWrapMode;
}

TextureFilterMode CubemapHandle::getFilterMode() const
{
    return mFilterMode;
}

int CubemapHandle::getWidth() const
{
    return mWidth;
}

CubemapHandle* CubemapHandle::create(int width, TextureFormat format, TextureWrapMode wrapMode,
    TextureFilterMode filterMode)
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLCubemapHandle(width, format, wrapMode, filterMode);
    case RenderAPI::DirectX:
        return new DirectXCubemapHandle(width, format, wrapMode, filterMode);
    }

    return nullptr;
}
