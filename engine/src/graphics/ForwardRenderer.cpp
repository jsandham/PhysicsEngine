#include "../../include/graphics/ForwardRenderer.h"
#include "../../include/graphics/ForwardRendererPasses.h"
#include "../../include/graphics/Graphics.h"

#include "../../include/components/Camera.h"
#include "../../include/components/MeshRenderer.h"
#include "../../include/components/Transform.h"

#include "../../include/core/Cubemap.h"
#include "../../include/core/Input.h"
#include "../../include/core/Log.h"
#include "../../include/core/Shader.h"
#include "../../include/core/Texture2D.h"
#include "../../include/core/Time.h"
#include "../../include/core/Util.h"

using namespace PhysicsEngine;

ForwardRenderer::ForwardRenderer()
{
}

ForwardRenderer::~ForwardRenderer()
{
}

void ForwardRenderer::init(World *world, bool renderToScreen)
{
    mWorld = world;
    mState.mRenderToScreen = renderToScreen;

    initializeRenderer(mWorld, mState);
}

void ForwardRenderer::update(const Input &input, Camera *camera,
                             const std::vector<std::pair<uint64_t, int>> &renderQueue,
                             const std::vector<RenderObject> &renderObjects,
                             const std::vector<SpriteObject> &spriteObjects)
{
    beginFrame(mWorld, camera, mState);

    if (camera->mSSAO == CameraSSAO::SSAO_On)
    {
        computeSSAO(mWorld, camera, mState, renderQueue, renderObjects);
    }

    for (size_t j = 0; j < mWorld->getNumberOfComponents<Light>(); j++)
    {
        Light *light = mWorld->getComponentByIndex<Light>(j);
        Transform *lightTransform = light->getComponent<Transform>(mWorld);

        renderShadows(mWorld, camera, light, lightTransform, mState, renderQueue, renderObjects);
        renderOpaques(mWorld, camera, light, lightTransform, mState, renderQueue, renderObjects);
        renderTransparents();
    }

    renderSprites(mWorld, camera, mState, spriteObjects);

    renderColorPicking(mWorld, camera, mState, renderQueue, renderObjects);

    postProcessing();

    endFrame(mWorld, camera, mState);
}