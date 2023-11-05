#ifndef DEFERREDRENDERER_H__
#define DEFERREDRENDERER_H__

#include <vector>

#include "../components/Camera.h"
#include "../core/Input.h"

#include "RenderObject.h"
#include "Renderer.h"
#include "RendererMeshes.h"
#include "RendererShaders.h"
#include "RendererUniforms.h"

namespace PhysicsEngine
{
class World;

class DeferredRenderer
{
  private:
    World *mWorld;

    QuadShader *mQuadShader;
    GBufferShader *mGBufferShader;
    ColorShader *mColorShader;
    ColorInstancedShader *mColorInstancedShader;

    CameraUniform *mCameraUniform;

    ScreenQuad *mScreenQuad;

  public:
    DeferredRenderer();
    ~DeferredRenderer();
    DeferredRenderer(const DeferredRenderer &other) = delete;
    DeferredRenderer &operator=(const DeferredRenderer &other) = delete;

    void init(World *world);
    void update(const Input &input, Camera *camera, const std::vector<DrawCallCommand> &commands,
                const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds);

  private:
    void beginDeferredFrame(Camera *camera);
    void geometryPass(Camera *camera, const std::vector<DrawCallCommand> &commands,
                      const std::vector<glm::mat4> &models);
    void lightingPass(Camera *camera, const std::vector<DrawCallCommand> &commands);
    void renderColorPickingDeferred(Camera *camera, const std::vector<DrawCallCommand> &commands,
                                    const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds);
    void endDeferredFrame(Camera *camera);
};

} // namespace PhysicsEngine

#endif