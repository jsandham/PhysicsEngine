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

void Renderer::bindFramebuffer(Framebuffer* fbo)
{
    return sInstance->bindFramebuffer_impl(fbo);
}

void Renderer::unbindFramebuffer()
{
    return sInstance->unbindFramebuffer_impl();
}

void Renderer::readColorAtPixel(Framebuffer *fbo, int x, int y, Color32 *color)
{
    return sInstance->readColorAtPixel_impl(fbo, x, y, color);
}

void Renderer::clearFrambufferColor(const Color &color)
{
    return sInstance->clearFrambufferColor_impl(color);
}

void Renderer::clearFrambufferColor(float r, float g, float b, float a)
{
    return sInstance->clearFrambufferColor_impl(r, g, b, a);
}

void Renderer::clearFramebufferDepth(float depth)
{
    return sInstance->clearFramebufferDepth_impl(depth);
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

void Renderer::createScreenQuad(unsigned int *vao, unsigned int *vbo)
{
    return sInstance->createScreenQuad_impl(vao, vbo);
}

void Renderer::renderScreenQuad(unsigned int vao)
{
    return sInstance->renderScreenQuad_impl(vao);
}