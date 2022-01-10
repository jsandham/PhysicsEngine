#ifndef FORWARDRENDERER_H__
#define FORWARDRENDERER_H__

#include <vector>

#include "../core/Input.h"

#include "Graphics.h"
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

    void init(World *world);
    void update(const Input &input, Camera *camera,
                const std::vector<RenderObject> &renderObjects,
                const std::vector<glm::mat4> &models, 
                const std::vector<Guid> &transformIds,
                const std::vector<SpriteObject> &spriteObjects);
};


void initializeRenderer(World *world, ForwardRendererState &state);

void beginFrame(World *world, Camera *camera, ForwardRendererState &state);

void computeSSAO(World *world, Camera *camera, ForwardRendererState &state,
                 const std::vector<RenderObject> &renderObjects, const std::vector<glm::mat4> &models);

void renderShadows(World *world, Camera *camera, Light *light, Transform *lightTransform, ForwardRendererState &state,
                   const std::vector<RenderObject> &renderObjects, const std::vector<glm::mat4> &models);

void renderOpaques(World *world, Camera *camera, Light *light, Transform *lightTransform, ForwardRendererState &state, 
                   const std::vector<RenderObject> &renderObjects, const std::vector<glm::mat4> &models);

void renderSprites(World *world, Camera *camera, ForwardRendererState &state,
                   const std::vector<SpriteObject> &spriteObjects);

void renderColorPicking(World *world, Camera *camera, ForwardRendererState &state,
                        const std::vector<RenderObject> &renderObjects, const std::vector<glm::mat4> &models,
                        const std::vector<Guid> &transformIds);

void renderTransparents();

void postProcessing();

void endFrame(World *world, Camera *camera, ForwardRendererState &state);

void calcCascadeOrthoProj(Camera *camera, glm::vec3 lightDirection, ForwardRendererState &state);
} // namespace PhysicsEngine

#endif