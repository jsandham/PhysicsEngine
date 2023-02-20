#ifndef RENDERER_API_H__
#define RENDERER_API_H__

#include <string>

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

#include "../components/Camera.h"
#include "../components/Light.h"
#include "../core/RenderTexture.h"
#include "../core/Shader.h"
#include "../core/Texture.h"

#include "GraphicsQuery.h"
#include "RenderObject.h"

#include "TextureHandle.h"
#include "VertexBuffer.h"
#include "UniformBuffer.h"

namespace PhysicsEngine
{
enum class Capability
{
    Depth_Testing,
    Blending,
    BackfaceCulling,
    LineSmoothing
};

enum class BlendingFactor
{
    ZERO,
    ONE,
    SRC_ALPHA,
    ONE_MINUS_SRC_ALPHA
};

class Renderer
{
private:
    static Renderer* sInstance;

public:
    static int INSTANCE_BATCH_SIZE;

    static void init();
    static Renderer *getRenderer();
    static void present();
    static void turnVsyncOn();
    static void turnVsyncOff();
    static void bindFramebuffer(Framebuffer* fbo);
    static void unbindFramebuffer();
    static void readColorAtPixel(Framebuffer *fbo, int x, int y, Color32 *color);
    static void clearFrambufferColor(const Color &color);
    static void clearFrambufferColor(float r, float g, float b, float a);
    static void clearFramebufferDepth(float depth);
    static void setViewport(int x, int y, int width, int height);
    static void turnOn(Capability capability);
    static void turnOff(Capability capability);
    static void setBlending(BlendingFactor source, BlendingFactor dest);
    static void draw(const RenderObject &renderObject, GraphicsQuery &query);
    static void drawInstanced(const RenderObject &renderObject, GraphicsQuery &query);

    static void beginQuery(unsigned int queryId);
    static void endQuery(unsigned int queryId, unsigned long long *elapsedTime);
    static void createScreenQuad(unsigned int *vao, unsigned int *vbo);
    static void renderScreenQuad(unsigned int vao);
    
protected:
    virtual void init_impl() = 0;
    virtual void present_impl() = 0;
    virtual void turnVsyncOn_impl() = 0;
    virtual void turnVsyncOff_impl() = 0;
    virtual void bindFramebuffer_impl(Framebuffer* fbo) = 0;
    virtual void unbindFramebuffer_impl() = 0;
    virtual void readColorAtPixel_impl(Framebuffer *fbo, int x, int y, Color32 *color) = 0;
    virtual void clearFrambufferColor_impl(const Color &color) = 0;
    virtual void clearFrambufferColor_impl(float r, float g, float b, float a) = 0;
    virtual void clearFramebufferDepth_impl(float depth) = 0;
    virtual void setViewport_impl(int x, int y, int width, int height) = 0;
    virtual void turnOn_impl(Capability capability) = 0;
    virtual void turnOff_impl(Capability capability) = 0;
    virtual void setBlending_impl(BlendingFactor source, BlendingFactor dest) = 0;
    virtual void draw_impl(const RenderObject &renderObject, GraphicsQuery &query) = 0;
    virtual void drawInstanced_impl(const RenderObject &renderObject, GraphicsQuery &query) = 0;

    virtual void beginQuery_impl(unsigned int queryId) = 0;
    virtual void endQuery_impl(unsigned int queryId, unsigned long long* elapsedTime) = 0;
    virtual void createScreenQuad_impl(unsigned int* vao, unsigned int* vbo) = 0;
    virtual void renderScreenQuad_impl(unsigned int vao) = 0;
};
}

#endif // RENDERER_API_H__