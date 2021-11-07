#ifndef DEFERREDRENDERER_H__
#define DEFERREDRENDERER_H__

#include <GL/glew.h>
#include <gl/gl.h>
#include <vector>

#include "../components/Camera.h"
#include "../core/Input.h"

//#include "DeferredRendererState.h"
#include "Graphics.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
class World;

class DeferredRenderer
{
  private:
    World *mWorld;

    DeferredRendererState mState;

  public:
    DeferredRenderer();
    ~DeferredRenderer();

    void init(World *world, bool renderToScreen);
    void update(const Input &input, Camera *camera, std::vector<RenderObject> &renderObjects);
};

void initializeDeferredRenderer(World *world, DeferredRendererState &state);

void beginDeferredFrame(World *world, Camera *camera, DeferredRendererState &state);

void geometryPass(World *world, Camera *camera, DeferredRendererState &state,
                  const std::vector<RenderObject> &renderObjects);

void lightingPass(World *world, Camera *camera, DeferredRendererState &state,
                  const std::vector<RenderObject> &renderObjects);

void endDeferredFrame(World *world, Camera *camera, DeferredRendererState &state);
} // namespace PhysicsEngine

#endif