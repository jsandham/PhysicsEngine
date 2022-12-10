#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/directx/DirectXFramebuffer.h"

#include <glm/glm.hpp>

using namespace PhysicsEngine;

DirectXFramebuffer::DirectXFramebuffer(int width, int height) : Framebuffer(width, height)
{
   
}

DirectXFramebuffer::DirectXFramebuffer(int width, int height, int numColorTex, bool addDepthTex)
    : Framebuffer(width, height, numColorTex, addDepthTex)
{

}

DirectXFramebuffer::~DirectXFramebuffer()
{
    
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