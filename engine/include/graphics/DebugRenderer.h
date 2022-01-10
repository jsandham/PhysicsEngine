#ifndef DEBUGRENDERER_H__
#define DEBUGRENDERER_H__

#include <vector>

#include "../core/Input.h"

#include "Graphics.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
class World;
class Camera;
class Transform;

class DebugRenderer
{
  private:
    World *mWorld;

    DebugRendererState mState;

  public:
    DebugRenderer();
    ~DebugRenderer();

    void init(World *world);
    void update(const Input &input, Camera *camera,
                const std::vector<RenderObject> &renderObjects,
                const std::vector<glm::mat4> &models,
                const std::vector<Guid> &transformIds);
};

void initializeDebugRenderer(World *world, DebugRendererState &state);

void beginDebugFrame(World *world, Camera *camera, DebugRendererState &state);

void renderDebug(World *world, Camera *camera, DebugRendererState &state,
                   const std::vector<RenderObject> &renderObjects,
                   const std::vector<glm::mat4> &models);

void renderDebugColorPicking(World *world, Camera *camera, DebugRendererState &state,
                             const std::vector<RenderObject> &renderObjects,
                             const std::vector<glm::mat4> &models,
                             const std::vector<Guid> &transformIds);

void endDebugFrame(World *world, Camera *camera, DebugRendererState &state);
} // namespace PhysicsEngine

#endif