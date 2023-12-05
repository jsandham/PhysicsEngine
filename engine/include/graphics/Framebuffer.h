#ifndef FRAMEBUFFER_H__
#define FRAMEBUFFER_H__

#include "../core/Color.h"

#include "RenderTextureHandle.h"

namespace PhysicsEngine
{
class Framebuffer
{
  protected:
    std::vector<RenderTextureHandle *> mColorTex;
    RenderTextureHandle *mDepthTex;

    unsigned int mWidth;
    unsigned int mHeight;
    unsigned int mNumColorTex;
    bool mAddDepthTex;

  public:
    Framebuffer(int width, int height);
    Framebuffer(int width, int height, int numColorTex, bool addDepthTex);
    Framebuffer(const Framebuffer &other) = delete;
    Framebuffer &operator=(const Framebuffer &other) = delete;
    virtual ~Framebuffer() = 0;

    int getWidth() const;
    int getHeight() const;

    virtual void clearColor(Color color) = 0;
    virtual void clearColor(float r, float g, float b, float a) = 0;
    virtual void clearDepth(float depth) = 0;
    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual void setViewport(int x, int y, int width, int height) = 0;
    virtual void setScissor(int x, int y, int width, int height) = 0;
    virtual void readColorAtPixel(int x, int y, Color32 *color) = 0;

    virtual RenderTextureHandle *getColorTex(size_t i = 0) = 0;
    virtual RenderTextureHandle *getDepthTex() = 0;

    static Framebuffer *create(int width, int height);
    static Framebuffer *create(int width, int height, int numColorTex, bool addDepthTex);
};
} // namespace PhysicsEngine

#endif