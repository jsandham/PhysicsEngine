#ifndef RENDERER_API_H__
#define RENDERER_API_H__

#include <string>

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

#include "../core/Color.h"

#include "GraphicsQuery.h"
#include "RenderObject.h"

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
    static Renderer *sInstance;

  public:
    static int INSTANCE_BATCH_SIZE;

    static void init();
    static Renderer *getRenderer();
    static void present();
    static void turnVsyncOn();
    static void turnVsyncOff();
    static void bindBackBuffer();
    static void unbindBackBuffer();
    static void clearBackBufferColor(const Color &color);
    static void clearBackBufferColor(float r, float g, float b, float a);
    static void setViewport(int x, int y, int width, int height);
    static void turnOn(Capability capability);
    static void turnOff(Capability capability);
    static void setBlending(BlendingFactor source, BlendingFactor dest);
    static void draw(MeshHandle *meshHandle, int start, int size, GraphicsQuery &query);
    static void drawIndexed(MeshHandle *meshHandle, int start, int size, GraphicsQuery &query);
    static void drawInstanced(MeshHandle *meshHandle, int start, int size, int instanceCount, GraphicsQuery &query);
    static void drawIndexedInstanced(MeshHandle *meshHandle, int start, int size, int instanceCount, GraphicsQuery &query);

    static void beginQuery(unsigned int queryId);
    static void endQuery(unsigned int queryId, unsigned long long *elapsedTime);

  protected:
    virtual void init_impl() = 0;
    virtual void present_impl() = 0;
    virtual void turnVsyncOn_impl() = 0;
    virtual void turnVsyncOff_impl() = 0;
    virtual void bindBackBuffer_impl() = 0;
    virtual void unbindBackBuffer_impl() = 0;
    virtual void clearBackBufferColor_impl(const Color &color) = 0;
    virtual void clearBackBufferColor_impl(float r, float g, float b, float a) = 0;
    virtual void setViewport_impl(int x, int y, int width, int height) = 0;
    virtual void turnOn_impl(Capability capability) = 0;
    virtual void turnOff_impl(Capability capability) = 0;
    virtual void setBlending_impl(BlendingFactor source, BlendingFactor dest) = 0;
    virtual void draw_impl(MeshHandle *meshHandle, int start, int size, GraphicsQuery &query) = 0;
    virtual void drawIndexed_impl(MeshHandle *meshHandle, int start, int size, GraphicsQuery &query) = 0;
    virtual void drawInstanced_impl(MeshHandle *meshHandle, int start, int size, int instanceCount, GraphicsQuery &query) = 0;
    virtual void drawIndexedInstanced_impl(MeshHandle *meshHandle, int start, int size, int instanceCount, GraphicsQuery &query) = 0;

    virtual void beginQuery_impl(unsigned int queryId) = 0;
    virtual void endQuery_impl(unsigned int queryId, unsigned long long *elapsedTime) = 0;
};
} // namespace PhysicsEngine

#endif // RENDERER_API_H__