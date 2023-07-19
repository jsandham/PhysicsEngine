#include "../../include/graphics/Framebuffer.h"
#include "../../include/graphics/RenderContext.h"

#include "../../include/graphics/platform/directx/DirectXFramebuffer.h"
#include "../../include/graphics/platform/opengl/OpenGLFramebuffer.h"

using namespace PhysicsEngine;

Framebuffer::Framebuffer(int width, int height) : mWidth(width), mHeight(height)
{
    mNumColorTex = 1;
    mAddDepthTex = true;
}

Framebuffer::Framebuffer(int width, int height, int numColorTex, bool addDepthTex)
    : mWidth(width), mHeight(height), mNumColorTex(numColorTex), mAddDepthTex(addDepthTex)
{
    mNumColorTex = 1;
    mAddDepthTex = true;
}

Framebuffer::~Framebuffer()
{
}

int Framebuffer::getWidth() const
{
    return mWidth;
}

int Framebuffer::getHeight() const
{
    return mHeight;
}

Framebuffer *Framebuffer::create(int width, int height)
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLFramebuffer(width, height);
    case RenderAPI::DirectX:
        return new DirectXFramebuffer(width, height);
    }

    return nullptr;
}

Framebuffer *Framebuffer::create(int width, int height, int numColorTex, bool addDepthTex)
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLFramebuffer(width, height, numColorTex, addDepthTex);
    case RenderAPI::DirectX:
        return new DirectXFramebuffer(width, height, numColorTex, addDepthTex);
    }

    return nullptr;
}