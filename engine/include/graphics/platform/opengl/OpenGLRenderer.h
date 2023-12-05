#ifndef OPENGL_RENDERER_API_H__
#define OPENGL_RENDERER_API_H__

#include "../../Renderer.h"
#include "OpenGLRenderContext.h"
#include "OpenGLTextureHandle.h"
#include "OpenGLVertexBuffer.h"

namespace PhysicsEngine
{
class OpenGLRenderer : public Renderer
{
  private:
    OpenGLRenderContext *mContext;

  protected:
    void init_impl() override;
    void present_impl() override;
    void turnVsyncOn_impl() override;
    void turnVsyncOff_impl() override;
    void bindBackBuffer_impl() override;
    void unbindBackBuffer_impl() override;
    void clearBackBufferColor_impl(const Color &color) override;
    void clearBackBufferColor_impl(float r, float g, float b, float a) override;
    void setViewport_impl(int x, int y, int width, int height) override;
    void setScissor_impl(int x, int y, int width, int height) override;
    void turnOn_impl(Capability capability) override;
    void turnOff_impl(Capability capability) override;
    void setBlending_impl(BlendingFactor source, BlendingFactor dest) override;
    void draw_impl(MeshHandle* meshHandle, size_t vertexOffset, size_t vertexCount, TimingQuery &query) override;
    void drawIndexed_impl(MeshHandle *meshHandle, size_t indexOffset, size_t indexCount, TimingQuery &query) override;
    void drawInstanced_impl(MeshHandle *meshHandle, size_t vertexOffset, size_t vertexCount, size_t instanceCount, TimingQuery &query) override;
    void drawIndexedInstanced_impl(MeshHandle *meshHandle, size_t indexOffset, size_t indexCount, size_t instanceCount, TimingQuery &query) override;

    void beginQuery_impl(unsigned int queryId) override;
    void endQuery_impl(unsigned int queryId, unsigned long long *elapsedTime) override;
};
} // namespace PhysicsEngine

#endif