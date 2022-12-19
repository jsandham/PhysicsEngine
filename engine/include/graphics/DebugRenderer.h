#ifndef DEBUGRENDERER_H__
#define DEBUGRENDERER_H__

#include <vector>

#include "../core/Input.h"

#include "Renderer.h"
#include "RendererShaders.h"
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
    DebugRenderer(const DebugRenderer& other) = delete;
    DebugRenderer& operator=(const DebugRenderer& other) = delete;

    void init(World *world);
    void update(const Input &input, Camera *camera,
                const std::vector<RenderObject> &renderObjects,
                const std::vector<glm::mat4> &models,
                const std::vector<Id> &transformIds);

  private:
    void initializeDebugRenderer();
    void beginDebugFrame(Camera *camera);
    void renderDebug(Camera *camera, const std::vector<RenderObject> &renderObjects, const std::vector<glm::mat4> &models);
    void renderDebugColorPicking(Camera *camera, const std::vector<RenderObject> &renderObjects,
                                        const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds);
    void endDebugFrame(Camera *camera);
};

} // namespace PhysicsEngine

#endif