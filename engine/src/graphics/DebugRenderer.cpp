#include "../../include/graphics/DebugRenderer.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

DebugRenderer::DebugRenderer()
{
}

DebugRenderer::~DebugRenderer()
{
}

void DebugRenderer::init(World *world)
{
    mWorld = world;

    initializeDebugRenderer(mWorld, mState);
}

void DebugRenderer::update(const Input &input, Camera *camera,
    const std::vector<std::pair<uint64_t, int>>& renderQueue,
    const std::vector<RenderObject>& renderObjects)
{
    beginDebugFrame(mWorld, camera, mState);

    renderDebug(mWorld, camera, mState, renderQueue, renderObjects);

    renderDebugColorPicking(mWorld, camera, mState, renderQueue, renderObjects);

    endDebugFrame(mWorld, camera, mState);
}

void PhysicsEngine::initializeDebugRenderer(World *world, DebugRendererState &state)
{
    Graphics::compileNormalShader(state);
    Graphics::compilePositionShader(state);
    Graphics::compileLinearDepthShader(state);
    Graphics::compileColorShader(state);
    Graphics::compileScreenQuadShader(state);

    Graphics::createScreenQuad(&state.mQuadVAO, &state.mQuadVBO);

    Graphics::createGlobalCameraUniforms(state.mCameraState);

    Graphics::turnOn(Capability::Depth_Testing);
}

void PhysicsEngine::beginDebugFrame(World *world, Camera *camera, DebugRendererState &state)
{
    camera->beginQuery();

    state.mCameraState.mProjection = camera->getProjMatrix();
    state.mCameraState.mView = camera->getViewMatrix();
    state.mCameraState.mCameraPos = camera->getComponent<Transform>()->mPosition;

    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    // update camera state data
    Graphics::setGlobalCameraUniforms(state.mCameraState);

    if (camera->mRenderTextureId.isValid())
    {
        RenderTexture *renderTexture = world->getAssetById<RenderTexture>(camera->mRenderTextureId);
        if (renderTexture != nullptr)
        {
            Graphics::bindFramebuffer(renderTexture->getNativeGraphicsMainFBO());
        }
        else
        {
            Graphics::bindFramebuffer(camera->getNativeGraphicsMainFBO());
        }
    }
    else
    {
        Graphics::bindFramebuffer(camera->getNativeGraphicsMainFBO());
    }
    Graphics::clearFrambufferColor(camera->mBackgroundColor);
    Graphics::clearFramebufferDepth(1.0f);
    Graphics::unbindFramebuffer();

    Graphics::bindFramebuffer(camera->getNativeGraphicsColorPickingFBO());
    Graphics::clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
    Graphics::clearFramebufferDepth(1.0f);
    Graphics::unbindFramebuffer();
}

void PhysicsEngine::renderDebug(World* world, Camera* camera, DebugRendererState& state,
    const std::vector<std::pair<uint64_t, int>>& renderQueue,
    const std::vector<RenderObject>& renderObjects)
{
    if (camera->mRenderTextureId.isValid())
    {
        RenderTexture *renderTexture = world->getAssetById<RenderTexture>(camera->mRenderTextureId);
        if (renderTexture != nullptr)
        {
            Graphics::bindFramebuffer(renderTexture->getNativeGraphicsMainFBO());
        }
        else
        {
            Graphics::bindFramebuffer(camera->getNativeGraphicsMainFBO());
        }
    }
    else
    {
        Graphics::bindFramebuffer(camera->getNativeGraphicsMainFBO());
    }

    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    int modelLoc = -1;

    switch (camera->mColorTarget)
    {
    case ColorTarget::Normal:
        Graphics::use(state.mNormalsShaderProgram);
        modelLoc = state.mNormalsShaderModelLoc;
        break;
    case ColorTarget::Position:
        Graphics::use(state.mPositionShaderProgram);
        modelLoc = state.mPositionShaderModelLoc;
        break;
    case ColorTarget::LinearDepth:
        Graphics::use(state.mLinearDepthShaderProgram);
        modelLoc = state.mLinearDepthShaderModelLoc;
        break;
    }

    assert(modelLoc != -1);

    for (size_t i = 0; i < renderQueue.size(); i++)
    {
        Graphics::setMat4(modelLoc, renderObjects[renderQueue[i].second].model);

        Graphics::render(renderObjects[renderQueue[i].second], camera->mQuery);
    }

    Graphics::unbindFramebuffer();
}

void PhysicsEngine::renderDebugColorPicking(World *world, Camera *camera, DebugRendererState &state,
                                       const std::vector<std::pair<uint64_t, int>> &renderQueue,
                                       const std::vector<RenderObject> &renderObjects)
{
    camera->clearColoring();

    // assign colors to render objects.
    uint32_t color = 1;
    for (size_t i = 0; i < renderQueue.size(); i++)
    {
        unsigned char r = static_cast<unsigned char>(255 - ((color & 0x000000FF) >> 0));
        unsigned char g = static_cast<unsigned char>(255 - ((color & 0x0000FF00) >> 8));
        unsigned char b = static_cast<unsigned char>(255 - ((color & 0x00FF0000) >> 16));
        unsigned char a = static_cast<unsigned char>(255);

        camera->assignColoring(Color32(r, g, b, a), renderObjects[renderQueue[i].second].transformId);

        color++;
    }

    Graphics::bindFramebuffer(camera->getNativeGraphicsColorPickingFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Graphics::use(state.mColorShaderProgram);

    color = 1;
    for (size_t i = 0; i < renderQueue.size(); i++)
    {
        unsigned char r = static_cast<unsigned char>(255 - ((color & 0x000000FF) >> 0));
        unsigned char g = static_cast<unsigned char>(255 - ((color & 0x0000FF00) >> 8));
        unsigned char b = static_cast<unsigned char>(255 - ((color & 0x00FF0000) >> 16));
        unsigned char a = static_cast<unsigned char>(255);

        color++;

        Graphics::setMat4(state.mColorShaderModelLoc, renderObjects[renderQueue[i].second].model);
        Graphics::setColor32(state.mColorShaderColorLoc, Color32(r, g, b, a));

        Graphics::render(renderObjects[renderQueue[i].second], camera->mQuery);
    }

    Graphics::unbindFramebuffer();

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::endDebugFrame(World *world, Camera *camera, DebugRendererState &state)
{
    if (camera->mRenderToScreen)
    {
        Graphics::bindFramebuffer(0);
        Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                              camera->getViewport().mHeight);

        Graphics::use(state.mQuadShaderProgram);
        Graphics::setTexture2D(state.mQuadShaderTexLoc, 0, camera->getNativeGraphicsColorTex());

        Graphics::renderScreenQuad(state.mQuadVAO);
        Graphics::unbindFramebuffer();
    }

    camera->endQuery();
}
