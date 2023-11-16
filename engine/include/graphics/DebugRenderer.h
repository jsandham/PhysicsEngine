#ifndef DEBUGRENDERER_H__
#define DEBUGRENDERER_H__

#include <vector>

#include "RenderObject.h"
#include "Renderer.h"
#include "RendererMeshes.h"
#include "RendererShaders.h"
#include "RendererUniforms.h"

namespace PhysicsEngine
{
class World;
class Camera;
class Transform;

class DebugRenderer
{
  private:
    World *mWorld;

    QuadShader *mQuadShader;
    NormalShader *mNormalsShader;
    PositionShader *mPositionShader;
    LinearDepthShader *mLinearDepthShader;
    ColorShader *mColorShader;
    NormalInstancedShader *mNormalsInstancedShader;
    PositionInstancedShader *mPositionInstancedShader;
    LinearDepthInstancedShader *mLinearDepthInstancedShader;
    ColorInstancedShader *mColorInstancedShader;

    CameraUniform *mCameraUniform;

    ScreenQuad *mScreenQuad;

  public:
    DebugRenderer();
    ~DebugRenderer();
    DebugRenderer(const DebugRenderer &other) = delete;
    DebugRenderer &operator=(const DebugRenderer &other) = delete;

    void init(World *world);
    void update(Camera *camera, const std::vector<DrawCallCommand> &commands,
                const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds);

  private:
    void beginDebugFrame(Camera *camera);
    void renderDebug(Camera *camera, const std::vector<DrawCallCommand> &commands,
                     const std::vector<glm::mat4> &models);
    void renderDebugColorPicking(Camera *camera, const std::vector<DrawCallCommand> &commands,
                                 const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds);
    void endDebugFrame(Camera *camera);
};

} // namespace PhysicsEngine

#endif