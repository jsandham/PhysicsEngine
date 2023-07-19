#ifndef RENDER_CONTEXT_OPENGL_H__
#define RENDER_CONTEXT_OPENGL_H__

#include "../../RenderContext.h"

#include <windows.h>

namespace PhysicsEngine
{
class OpenGLRenderContext : public RenderContext
{
  private:
    HGLRC mOpenGLRC;
    HDC mWindowDC;

  public:
    OpenGLRenderContext(void *window);
    ~OpenGLRenderContext();

    void present();
    void turnVsyncOn();
    void turnVsyncOff();
    void bindBackBuffer();
    void unBindBackBuffer();
    void clearBackBufferColor(float r, float g, float b, float a);

    static OpenGLRenderContext *get()
    {
        return (OpenGLRenderContext *)sContext;
    }

  private:
    static void SetSwapInterval(int interval);
};
} // namespace PhysicsEngine

#endif // RENDER_CONTEXT_OPENGL_H__