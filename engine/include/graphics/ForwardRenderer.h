#ifndef __FORWARDRENDERER_H__
#define __FORWARDRENDERER_H__

#include <GL/glew.h>
#include <gl/gl.h>
#include <vector>

#include "../components/Camera.h"
#include "../core/Input.h"

#include "ForwardRendererState.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
class World;

class ForwardRenderer
{
  private:
    World *mWorld;

    ForwardRendererState mState;

  public:
    ForwardRenderer();
    ~ForwardRenderer();

    void init(World *world, bool renderToScreen);
    void update(const Input &input, Camera *camera, const std::vector<std::pair<uint64_t, int>> &renderQueue,
                const std::vector<RenderObject> &renderObjects);
};
} // namespace PhysicsEngine

#endif