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

    QuadShader *mQuadShader;
    DepthShader *mDepthShader;
    DepthCubemapShader *mDepthCubemapShader;
    GeometryShader *mGeometryShader;
    ColorShader *mColorShader;
    ColorInstancedShader *mColorInstancedShader;
    SSAOShader *mSsaoShader;
    SpriteShader *mSpriteShader;

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

  private:
    void beginFrame(Camera* camera);
    void computeSSAO(Camera* camera, const std::vector<RenderObject>& renderObjects, const std::vector<glm::mat4>& models);
    void renderShadows(Camera* camera, Light* light, Transform* lightTransform, const std::vector<RenderObject>& renderObjects, const std::vector<glm::mat4>& models);
    void renderOpaques(Camera* camera, Light* light, Transform* lightTransform, const std::vector<RenderObject>& renderObjects, const std::vector<glm::mat4>& models);
    void renderSprites(Camera* camera, const std::vector<SpriteObject>& spriteObjects);
    void renderColorPicking(Camera* camera, const std::vector<RenderObject>& renderObjects, const std::vector<glm::mat4>& models, const std::vector<Id>& transformIds);
    void renderTransparents();
    void postProcessing();
    void endFrame(Camera* camera);
    void calcCascadeOrthoProj(Camera* camera, glm::vec3 lightDirection);

};

} // namespace PhysicsEngine

#endif