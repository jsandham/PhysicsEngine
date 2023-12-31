#ifndef FORWARDRENDERER_H__
#define FORWARDRENDERER_H__

#include <vector>

#include "../components/Light.h"

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

    CameraUniform *mCameraUniform;
    LightUniform *mLightUniform;

    ScreenQuad *mScreenQuad;

    // directional light cascade shadow map data
    float mCascadeEnds[6];
    glm::mat4 mCascadeOrthoProj[5];
    glm::mat4 mCascadeLightView[5];

    // spotlight shadow map data
    glm::mat4 mShadowViewMatrix;
    glm::mat4 mShadowProjMatrix;

    // pointlight cubemap shadow map data
    glm::mat4 mCubeViewProjMatrices[6];

  public:
    ForwardRenderer();
    ~ForwardRenderer();
    ForwardRenderer(const ForwardRenderer &other) = delete;
    ForwardRenderer &operator=(const ForwardRenderer &other) = delete;

    void init(World *world);
    void update(Camera *camera, const std::vector<DrawCallCommand> &commands,
                const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds, OcclusionQuery &occlusionQuery);

  private:
    void beginFrame(Camera *camera);
    void computeSSAO(Camera *camera, const std::vector<DrawCallCommand> &commands,
                     const std::vector<glm::mat4> &models);
    void renderShadows(Camera *camera, Light *light, Transform *lightTransform, const std::vector<DrawCallCommand> &commands,
                       const std::vector<glm::mat4> &models);
    void renderOpaques(Camera *camera, Light *light, Transform *lightTransform, const std::vector<DrawCallCommand> &commands,
                       const std::vector<glm::mat4> &models,
                       OcclusionQuery &occlusionQuery);
    void renderColorPicking(Camera *camera, const std::vector<DrawCallCommand> &commands,
                            const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds);
    void renderTransparents();
    void postProcessing();
    void endFrame(Camera *camera);
    void calcCascadeOrthoProj(Camera *camera, glm::vec3 lightDirection);
    void renderDirLightShadowMap(Camera *camera, Light *light, Transform *lightTransform,
                                         const std::vector<DrawCallCommand> &commands,
                                         const std::vector<glm::mat4> &models);
    void renderSpotLightShadowMap(Camera *camera, Light *light, Transform *lightTransform,
                                         const std::vector<DrawCallCommand> &commands,
                                         const std::vector<glm::mat4> &models);
    void renderPointLightShadowMap(Camera *camera, Light *light, Transform *lightTransform,
                                         const std::vector<DrawCallCommand> &commands,
                                         const std::vector<glm::mat4> &models);

};

} // namespace PhysicsEngine

#endif