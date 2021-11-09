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
    void update(const Input &input, Camera *camera, const std::vector<std::pair<uint64_t, int>> &renderQueue,
                const std::vector<RenderObject> &renderObjects, const std::vector<SpriteObject> &spriteObjects);
};


void initializeRenderer(World *world, ForwardRendererState &state);

void beginFrame(World *world, Camera *camera, ForwardRendererState &state);

void computeSSAO(World *world, Camera *camera, ForwardRendererState &state,
                 const std::vector<std::pair<uint64_t, int>> &renderQueue,
                 const std::vector<RenderObject> &renderObjects);

void renderShadows(World *world, Camera *camera, Light *light, Transform *lightTransform, ForwardRendererState &state,
                   const std::vector<std::pair<uint64_t, int>> &renderQueue,
                   const std::vector<RenderObject> &renderObjects);

void renderOpaques(World *world, Camera *camera, Light *light, Transform *lightTransform, ForwardRendererState &state,
                   const std::vector<std::pair<uint64_t, int>> &renderQueue,
                   const std::vector<RenderObject> &renderObjects);

void renderSprites(World *world, Camera *camera, ForwardRendererState &state,
                   const std::vector<SpriteObject> &spriteObjects);

void renderColorPicking(World *world, Camera *camera, ForwardRendererState &state,
                        const std::vector<std::pair<uint64_t, int>> &renderQueue,
                        const std::vector<RenderObject> &renderObjects);

void renderTransparents();

void postProcessing();

void endFrame(World *world, Camera *camera, ForwardRendererState &state);

void calcCascadeOrthoProj(Camera *camera, glm::vec3 lightDirection, ForwardRendererState &state);
} // namespace PhysicsEngine

#endif