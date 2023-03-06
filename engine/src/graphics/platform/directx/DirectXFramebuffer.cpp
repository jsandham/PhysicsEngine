#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/directx/DirectXFramebuffer.h"

#include <glm/glm.hpp>

using namespace PhysicsEngine;

DirectXFramebuffer::DirectXFramebuffer(int width, int height) : Framebuffer(width, height)
{
    mColorTex.resize(1);
   
    mColorTex[0] =
        TextureHandle::create(mWidth, mHeight, TextureFormat::RGBA, TextureWrapMode::ClampToEdge, TextureFilterMode::Nearest);
    mDepthTex = TextureHandle::create(mWidth, mHeight, TextureFormat::Depth, TextureWrapMode::ClampToEdge,
                                      TextureFilterMode::Nearest);
}

DirectXFramebuffer::DirectXFramebuffer(int width, int height, int numColorTex, bool addDepthTex)
    : Framebuffer(width, height, numColorTex, addDepthTex)
{
    mColorTex.resize(mNumColorTex);

    for (size_t i = 0; i < mColorTex.size(); i++)
    {
        mColorTex[i] = TextureHandle::create(mWidth, mHeight, TextureFormat::RGBA, TextureWrapMode::ClampToEdge,
                                             TextureFilterMode::Nearest);
    }

    if (mAddDepthTex)
    {
        mDepthTex = TextureHandle::create(mWidth, mHeight, TextureFormat::Depth, TextureWrapMode::ClampToEdge,
                                          TextureFilterMode::Nearest);
    }
    else
    {
        mDepthTex = nullptr;
    }
}

DirectXFramebuffer::~DirectXFramebuffer()
{
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

void DirectXFramebuffer::clearColor(Color color)
{

}

void DirectXFramebuffer::clearColor(float r, float g, float b, float a)
{

}

void DirectXFramebuffer::clearDepth(float depth)
{

}

void DirectXFramebuffer::bind()
{
    
}

void DirectXFramebuffer::unbind()
{
    
}

void DirectXFramebuffer::setViewport(int x, int y, int width, int height)
{
    assert(x >= 0);
    assert(y >= 0);
    assert((unsigned int)(x + width) <= mWidth);
    assert((unsigned int)(y + height) <= mHeight);
}

TextureHandle *DirectXFramebuffer::getColorTex(size_t i)
{
    assert(i < mColorTex.size());
    return mColorTex[i];
}

TextureHandle *DirectXFramebuffer::getDepthTex()
{
    return mDepthTex;
}

void *DirectXFramebuffer::getHandle()
{
    return nullptr;
}