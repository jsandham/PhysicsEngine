#include "../../include/graphics/DeferredRenderer.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

DeferredRenderer::DeferredRenderer()
{
}

DeferredRenderer::~DeferredRenderer()
{
}

void DeferredRenderer::init(World *world)
{
    mWorld = world;

    initializeDeferredRenderer(mWorld, mState);
}

void DeferredRenderer::update(const Input &input, Camera *camera,
                              const std::vector<std::pair<uint64_t, int>> &renderQueue,
                              const std::vector<RenderObject> &renderObjects)
{
    beginDeferredFrame(mWorld, camera, mState);

    geometryPass(mWorld, camera, mState, renderObjects);
    lightingPass(mWorld, camera, mState, renderObjects);

    renderColorPickingDeferred(mWorld, camera, mState, renderQueue, renderObjects);

    endDeferredFrame(mWorld, camera, mState);
}

void PhysicsEngine::initializeDeferredRenderer(World *world, DeferredRendererState &state)
{
    Graphics::compileScreenQuadShader(state);
    Graphics::compileGBufferShader(state);
    Graphics::compileColorShader(state);

    Graphics::createScreenQuad(&state.mQuadVAO, &state.mQuadVBO);

    Graphics::createGlobalCameraUniforms(state.mCameraState);

    Graphics::turnOn(Capability::Depth_Testing);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::beginDeferredFrame(World *world, Camera *camera, DeferredRendererState &state)
{
    camera->beginQuery();

    state.mCameraState.mProjection = camera->getProjMatrix();
    state.mCameraState.mView = camera->getViewMatrix();
    state.mCameraState.mCameraPos = camera->getComponent<Transform>()->mPosition;

    // set camera state binding point and update camera state data
    Graphics::setGlobalCameraUniforms(state.mCameraState);

    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

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

    Graphics::bindFramebuffer(camera->getNativeGraphicsGeometryFBO());
    Graphics::clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
    Graphics::clearFramebufferDepth(1.0f);
    Graphics::unbindFramebuffer();

    Graphics::bindFramebuffer(camera->getNativeGraphicsColorPickingFBO());
    Graphics::clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
    Graphics::clearFramebufferDepth(1.0f);
    Graphics::unbindFramebuffer();
}

void PhysicsEngine::geometryPass(World *world, Camera *camera, DeferredRendererState &state,
                                 const std::vector<RenderObject> &renderObjects)
{
    // fill geometry framebuffer
    Graphics::bindFramebuffer(camera->getNativeGraphicsGeometryFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);
    
    Graphics::use(state.mGBufferShaderProgram);
    for (size_t i = 0; i < renderObjects.size(); i++)
    {
        Graphics::setMat4(state.mGBufferShaderModelLoc, renderObjects[i].model);
        Graphics::render(renderObjects[i], camera->mQuery);
    }
    Graphics::unbindFramebuffer();

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::lightingPass(World *world, Camera *camera, DeferredRendererState &state,
                                 const std::vector<RenderObject> &renderObjects)
{
    Graphics::bindFramebuffer(camera->getNativeGraphicsMainFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    //Graphics::use(state.mSimpleLitDeferredShaderProgram);

    //Graphics::setTexture2D(state.mPositionTexLoc, 0, camera->getNativeGraphicsPositionTex());
    //Graphics::setTexture2D(state.mNormalTexLoc, 1, camera->getNativeGraphicsNormalTex());
    //Graphics::setTexture2D(state.mAlbedoSpecTexLoc, 2, camera->getNativeGraphicsAlbedoSpecTex());

    for (size_t i = 0; i < world->getNumberOfComponents<Light>(); i++)
    {
        Light *light = world->getComponentByIndex<Light>(i);
        Transform *lightTransform = light->getComponent<Transform>();
        if (lightTransform != nullptr)
        {
            //Graphics::setVec3(state->mSimpleLitDeferredShaderLightPosLocs, lightTransform->mPosition);
            //Graphics::setVec3(state->mSimpleLitDeferredShaderLightColLocs, light->mAmbient);
        }
    }

    //glBindVertexArray(state.mQuadVAO);
    //glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    //glBindVertexArray(0);

    Graphics::unbindFramebuffer();
}

void PhysicsEngine::renderColorPickingDeferred(World *world, Camera *camera, DeferredRendererState &state,
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

void PhysicsEngine::endDeferredFrame(World *world, Camera *camera, DeferredRendererState &state)
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