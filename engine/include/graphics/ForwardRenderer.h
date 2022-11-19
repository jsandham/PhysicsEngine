#ifndef FORWARDRENDERER_H__
#define FORWARDRENDERER_H__

#include <vector>

#include "../core/Input.h"

#include "Renderer.h"
#include "RendererShaders.h"
#include "RenderObject.h"
#include "SpriteObject.h"

namespace PhysicsEngine
{
class World;
class Camera;
class Transform;

class ForwardRenderer
{
  private:
    World *mWorld;

    ForwardRendererState mState;

  public:
    ForwardRenderer();
    ~ForwardRenderer();
    ForwardRenderer(const ForwardRenderer& other) = delete;
    ForwardRenderer& operator=(const ForwardRenderer& other) = delete;

    void init(World *world);
    void update(const Input &input, Camera *camera,
                const std::vector<RenderObject> &renderObjects,
                const std::vector<glm::mat4> &models, 
                const std::vector<Id> &transformIds,
                const std::vector<SpriteObject> &spriteObjects);
};

} // namespace PhysicsEngine

#endif