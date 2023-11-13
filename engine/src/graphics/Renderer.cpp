#include "../../include/graphics/Renderer.h"
#include "../../include/graphics/RenderContext.h"
#include "../../include/graphics/RendererMeshes.h"
#include "../../include/graphics/RendererShaders.h"
#include "../../include/graphics/RendererUniforms.h"

#include "../../include/graphics/platform/directx/DirectXRenderer.h"
#include "../../include/graphics/platform/opengl/OpenGLRenderer.h"

using namespace PhysicsEngine;

int Renderer::INSTANCE_BATCH_SIZE = 100;
int Renderer::MAX_OCCLUDER_COUNT = 20;
int Renderer::MAX_OCCLUDER_VERTEX_COUNT = 10000;
int Renderer::MAX_OCCLUDER_INDEX_COUNT = 5000;

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

    RendererShaders::createInternalShaders();
    RendererUniforms::createInternalUniforms();
    RendererMeshes::createInternalMeshes();
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

void Renderer::draw(MeshHandle *meshHandle, size_t vertexOffset, size_t vertexCount, TimingQuery &query)
{
    return sInstance->draw_impl(meshHandle, vertexOffset, vertexCount, query);
}

void Renderer::drawIndexed(MeshHandle *meshHandle, size_t indexOffset, size_t indexCount, TimingQuery &query)
{
    return sInstance->drawIndexed_impl(meshHandle, indexOffset, indexCount, query);
}

void Renderer::drawInstanced(MeshHandle *meshHandle, size_t vertexOffset, size_t vertexCount, size_t instanceCount, TimingQuery &query)
{
    return sInstance->drawInstanced_impl(meshHandle, vertexOffset, vertexCount, instanceCount, query);
}

void Renderer::drawIndexedInstanced(MeshHandle *meshHandle, size_t indexOffset, size_t indexCount, size_t instanceCount,
                                    TimingQuery &query)
{
    return sInstance->drawIndexedInstanced_impl(meshHandle, indexOffset, indexCount, instanceCount, query);
}

void Renderer::beginQuery(unsigned int queryId)
{
    return sInstance->beginQuery_impl(queryId);
}

void Renderer::endQuery(unsigned int queryId, unsigned long long *elapsedTime)
{
    return sInstance->endQuery_impl(queryId, elapsedTime);
}