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
        OpenGLRenderContext* mContext;

      protected:
        void init_impl() override;
        void present_impl() override;
        void turnVsyncOn_impl() override;
        void turnVsyncOff_impl() override;
        void bindFramebuffer_impl(Framebuffer* fbo) override;
        void unbindFramebuffer_impl() override;
        void readColorAtPixel_impl(Framebuffer *fbo, int x, int y, Color32 *color) override;
        void clearFrambufferColor_impl(const Color &color) override;
        void clearFrambufferColor_impl(float r, float g, float b, float a) override;
        void clearFramebufferDepth_impl(float depth) override;
        void setViewport_impl(int x, int y, int width, int height) override;
        void turnOn_impl(Capability capability) override;
        void turnOff_impl(Capability capability) override;
        void setBlending_impl(BlendingFactor source, BlendingFactor dest) override;
        void draw_impl(const RenderObject &renderObject, GraphicsQuery &query) override;
        void drawInstanced_impl(const RenderObject &renderObject, GraphicsQuery &query) override;

        void beginQuery_impl(unsigned int queryId) override;
        void endQuery_impl(unsigned int queryId, unsigned long long *elapsedTime) override;
        void createScreenQuad_impl(unsigned int *vao, unsigned int *vbo) override;
        void renderScreenQuad_impl(unsigned int vao) override;
	};
}

#endif