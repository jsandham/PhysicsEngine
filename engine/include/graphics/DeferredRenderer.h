#ifndef DEFERREDRENDERER_H__
#define DEFERREDRENDERER_H__

#include <vector>

#include "../components/Camera.h"
#include "../core/Input.h"

#include "Renderer.h"
#include "RendererShaders.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
class World;

class DeferredRenderer
{
  private:
    World *mWorld;

    QuadShader* mQuadShader;
    GBufferShader* mGBufferShader;
    ColorShader* mColorShader;
    ColorInstancedShader* mColorInstancedShader;

    DeferredRendererState mState;

  public:
    DeferredRenderer();
    ~DeferredRenderer();
    DeferredRenderer(const DeferredRenderer& other) = delete;
    DeferredRenderer& operator=(const DeferredRenderer& other) = delete;

    void init(World *world);
    void update(const Input &input, Camera *camera,
                const std::vector<RenderObject> &renderObjects,
                const std::vector<glm::mat4> &models,
                const std::vector<Id> &transformIds);

  private:
    void beginDeferredFrame(Camera *camera);
    void geometryPass(Camera *camera, const std::vector<RenderObject> &renderObjects, const std::vector<glm::mat4> &models);
    void lightingPass(Camera *camera, const std::vector<RenderObject> &renderObjects);
    void renderColorPickingDeferred(Camera *camera, const std::vector<RenderObject> &renderObjects,
                                           const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds);
    void endDeferredFrame(Camera *camera);
};

} // namespace PhysicsEngine

#endif