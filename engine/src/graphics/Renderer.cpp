#include "../../include/graphics/Renderer.h"
#include "../../include/graphics/RenderContext.h"

#include "../../include/graphics/platform/opengl/OpenGLRenderer.h"
#include "../../include/graphics/platform/directx/DirectXRenderer.h"

using namespace PhysicsEngine;

int Renderer::INSTANCE_BATCH_SIZE = 1000;
Renderer *Renderer::sInstance = nullptr;

void Renderer::init()
{
    switch (RenderContext::getRenderAPI())
    {
        case RenderAPI::OpenGL:
            sInstance = new OpenGLRenderer();
            break;
        case RenderAPI::DirectX:
            sInstance = new DirectXRenderer();
            break;
    }

    sInstance->init_impl();
}

Renderer *Renderer::getRenderer()
{
    return sInstance;
}

void Renderer::present()
{
    sInstance->present_impl();
}

void Renderer::turnVsyncOn()
{
    sInstance->turnVsyncOn_impl();
}

void Renderer::turnVsyncOff()
{
    sInstance->turnVsyncOff_impl();
}

void Renderer::bindBackBuffer()
{
    return sInstance->bindBackBuffer_impl();
}

void Renderer::unbindBackBuffer()
{
    return sInstance->unbindBackBuffer_impl();
}

void Renderer::clearBackBufferColor(const Color &color)
{
     return sInstance->clearBackBufferColor_impl(color);
}

void Renderer::clearBackBufferColor(float r, float g, float b, float a)
{
     return sInstance->clearBackBufferColor_impl(r, g, b, a);
}

void Renderer::setViewport(int x, int y, int width, int height)
{
    return sInstance->setViewport_impl(x, y, width, height);
}
    
void Renderer::turnOn(Capability capability)
{
    return sInstance->turnOn_impl(capability);
}

void Renderer::turnOff(Capability capability)
{
    return sInstance->turnOff_impl(capability);
}

void Renderer::setBlending(BlendingFactor source, BlendingFactor dest)
{
    return sInstance->setBlending_impl(source, dest);
}

void Renderer::draw(const RenderObject &renderObject, GraphicsQuery &query)
{
    return sInstance->draw_impl(renderObject, query);
}

void Renderer::drawIndexed(const RenderObject &renderObject, GraphicsQuery &query)
{
    return sInstance->drawIndexed_impl(renderObject, query);
}

void Renderer::drawInstanced(const RenderObject &renderObject, GraphicsQuery &query)
{
    return sInstance->drawInstanced_impl(renderObject, query);
}

void Renderer::drawIndexedInstanced(const RenderObject &renderObject, GraphicsQuery &query)
{
    return sInstance->drawIndexedInstanced_impl(renderObject, query);
}

void Renderer::beginQuery(unsigned int queryId)
{
    return sInstance->beginQuery_impl(queryId);
}

void Renderer::endQuery(unsigned int queryId, unsigned long long *elapsedTime)
{
    return sInstance->endQuery_impl(queryId, elapsedTime);
}