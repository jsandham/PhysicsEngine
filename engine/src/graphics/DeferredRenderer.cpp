#include "../../include/graphics/DeferredRenderer.h"
#include "../../include/core/World.h"

#include "../../include/core/Log.h"

using namespace PhysicsEngine;

static void initializeDeferredRenderer(World* world, DeferredRendererState& state);
static void beginDeferredFrame(World* world, Camera* camera, DeferredRendererState& state);
static void geometryPass(World* world, Camera* camera, DeferredRendererState& state,
    const std::vector<RenderObject>& renderObjects,
    const std::vector<glm::mat4>& models);
static void lightingPass(World* world, Camera* camera, DeferredRendererState& state,
    const std::vector<RenderObject>& renderObjects);
static void renderColorPickingDeferred(World* world, Camera* camera, DeferredRendererState& state,
    const std::vector<RenderObject>& renderObjects,
    const std::vector<glm::mat4>& models,
    const std::vector<Id>& transformIds);
static void endDeferredFrame(World* world, Camera* camera, DeferredRendererState& state);

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
                              const std::vector<RenderObject> &renderObjects,
                              const std::vector<glm::mat4> &models,
                              const std::vector<Id> &transformIds)
{
    beginDeferredFrame(mWorld, camera, mState);

    geometryPass(mWorld, camera, mState, renderObjects, models);
    lightingPass(mWorld, camera, mState, renderObjects);

    renderColorPickingDeferred(mWorld, camera, mState, renderObjects, models, transformIds);

    endDeferredFrame(mWorld, camera, mState);
}

static void initializeDeferredRenderer(World *world, DeferredRendererState &state)
{
    state.mGBufferShaderProgram = RendererShaders::getRendererShaders()->getGBufferShader().mProgram;
    state.mGBufferShaderModelLoc = RendererShaders::getRendererShaders()->getGBufferShader().mModelLoc;
    state.mGBufferShaderDiffuseTexLoc = RendererShaders::getRendererShaders()->getGBufferShader().mDiffuseTexLoc;
    state.mGBufferShaderSpecTexLoc = RendererShaders::getRendererShaders()->getGBufferShader().mSpecTexLoc;

    state.mColorShaderProgram = RendererShaders::getRendererShaders()->getColorShader().mProgram;
    state.mColorShaderModelLoc = RendererShaders::getRendererShaders()->getColorShader().mModelLoc;
    state.mColorShaderColorLoc = RendererShaders::getRendererShaders()->getColorShader().mColorLoc;
    state.mColorInstancedShaderProgram = RendererShaders::getRendererShaders()->getColorInstancedShader().mProgram;

    state.mQuadShaderProgram = RendererShaders::getRendererShaders()->getScreenQuadShader().mProgram;
    state.mQuadShaderTexLoc = RendererShaders::getRendererShaders()->getScreenQuadShader().mTexLoc;

    Renderer::getRenderer()->createScreenQuad(&state.mQuadVAO, &state.mQuadVBO);

    Renderer::getRenderer()->createGlobalCameraUniforms(state.mCameraState);

    Renderer::getRenderer()->turnOn(Capability::Depth_Testing);
}

static void beginDeferredFrame(World *world, Camera *camera, DeferredRendererState &state)
{
    camera->beginQuery();

    state.mCameraState.mProjection = camera->getProjMatrix();
    state.mCameraState.mView = camera->getViewMatrix();
    state.mCameraState.mViewProjection = camera->getProjMatrix() * camera->getViewMatrix();
    state.mCameraState.mCameraPos = camera->getComponent<Transform>()->getPosition();

    // set camera state binding point and update camera state data
    Renderer::getRenderer()->setGlobalCameraUniforms(state.mCameraState);

    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Framebuffer *framebuffer = nullptr;

    if (camera->mRenderTextureId.isValid())
    {
        RenderTexture *renderTexture = world->getAssetByGuid<RenderTexture>(camera->mRenderTextureId);
        if (renderTexture != nullptr)
        {
            //Renderer::getRenderer()->bindFramebuffer(renderTexture->getNativeGraphicsMainFBO());
            framebuffer = renderTexture->getNativeGraphicsMainFBO();
        }
        else
        {
            //Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
            framebuffer = camera->getNativeGraphicsMainFBO();
        }
    }
    else
    {
        //Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
        framebuffer = camera->getNativeGraphicsMainFBO();
    }

    //Renderer::getRenderer()->clearFrambufferColor(camera->mBackgroundColor);
    //Renderer::getRenderer()->clearFramebufferDepth(1.0f);
    //Renderer::getRenderer()->unbindFramebuffer();
    framebuffer->bind();
    framebuffer->clearColor(Color::black);
    framebuffer->clearDepth(1.0f);
    framebuffer->unbind();

    //Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsGeometryFBO());
    //Renderer::getRenderer()->clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
    //Renderer::getRenderer()->clearFramebufferDepth(1.0f);
    //Renderer::getRenderer()->unbindFramebuffer();
    camera->getNativeGraphicsGeometryFBO()->bind();
    camera->getNativeGraphicsGeometryFBO()->clearColor(Color::black);
    camera->getNativeGraphicsGeometryFBO()->clearDepth(1.0f);
    camera->getNativeGraphicsGeometryFBO()->unbind();

    //Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsColorPickingFBO());
    //Renderer::getRenderer()->clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
    //Renderer::getRenderer()->clearFramebufferDepth(1.0f);
    //Renderer::getRenderer()->unbindFramebuffer();
    camera->getNativeGraphicsColorPickingFBO()->bind();
    camera->getNativeGraphicsColorPickingFBO()->clearColor(Color::black);
    camera->getNativeGraphicsColorPickingFBO()->clearDepth(1.0f);
    camera->getNativeGraphicsColorPickingFBO()->unbind();
}

static void geometryPass(World *world, Camera *camera, DeferredRendererState &state,
                                 const std::vector<RenderObject> &renderObjects,
                                 const std::vector<glm::mat4> &models)
{
    // fill geometry framebuffer
    //Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsGeometryFBO());
    camera->getNativeGraphicsGeometryFBO()->bind();
    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);
    
    Renderer::getRenderer()->use(state.mGBufferShaderProgram);

    int mGBufferShaderDiffuseTexLoc = Renderer::getRenderer()->findUniformLocation("texture_diffuse1", state.mGBufferShaderProgram);
    int mGBufferShaderSpecTexLoc = Renderer::getRenderer()->findUniformLocation("texture_specular1", state.mGBufferShaderProgram);

    //std::string message = "mGBufferShaderDiffuseTexLoc: " + std::to_string(mGBufferShaderDiffuseTexLoc) +
    //                      " mGBufferShaderSpecTexLoc: " + std::to_string(mGBufferShaderSpecTexLoc) + "\n";
    //Log::info(message.c_str());

    //for (size_t i = 0; i < renderObjects.size(); i++)
    //{
    //    Renderer::getRenderer()->setMat4(state.mGBufferShaderModelLoc, renderObjects[i].model);
    //    Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);
    //}
    //int currentMaterialIndex = -1;
    //Material *material = nullptr;

    int modelIndex = 0;
    for (size_t i = 0; i < renderObjects.size(); i++)
    {
        if (renderObjects[i].instanced)
        {
            modelIndex += renderObjects[i].instanceCount;
        }
        else
        {
            Renderer::getRenderer()->setMat4(state.mGBufferShaderModelLoc, models[modelIndex]);

            // if (currentMaterialIndex != renderObjects[renderQueue[i].second].materialIndex)
            //{
            //    material = world->getAssetByIndex<Material>(renderObjects[renderQueue[i].second].materialIndex);
            //    material->apply(world);
            //
            //    currentMaterialIndex = renderObjects[renderQueue[i].second].materialIndex;
            //}

            Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);
            modelIndex++;
        }
    }

    //Renderer::getRenderer()->unbindFramebuffer();
    camera->getNativeGraphicsGeometryFBO()->unbind();
}

static void lightingPass(World *world, Camera *camera, DeferredRendererState &state,
                                 const std::vector<RenderObject> &renderObjects)
{
    //Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
    camera->getNativeGraphicsMainFBO()->bind();
    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    //Renderer::getRenderer()->use(state.mSimpleLitDeferredShaderProgram);

    //Renderer::getRenderer()->setTexture2D(state.mPositionTexLoc, 0, camera->getNativeGraphicsPositionTex());
    //Renderer::getRenderer()->setTexture2D(state.mNormalTexLoc, 1, camera->getNativeGraphicsNormalTex());
    //Renderer::getRenderer()->setTexture2D(state.mAlbedoSpecTexLoc, 2, camera->getNativeGraphicsAlbedoSpecTex());

    for (size_t i = 0; i < world->getActiveScene()->getNumberOfComponents<Light>(); i++)
    {
        Light *light = world->getActiveScene()->getComponentByIndex<Light>(i);
        Transform *lightTransform = light->getComponent<Transform>();
        if (lightTransform != nullptr)
        {
            //Renderer::getRenderer()->setVec3(state->mSimpleLitDeferredShaderLightPosLocs, lightTransform->mPosition);
            //Renderer::getRenderer()->setVec3(state->mSimpleLitDeferredShaderLightColLocs, light->mAmbient);
        }
    }

    //glBindVertexArray(state.mQuadVAO);
    //glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    //glBindVertexArray(0);

    //Renderer::getRenderer()->unbindFramebuffer();
    camera->getNativeGraphicsMainFBO()->unbind();
}

static void renderColorPickingDeferred(World *world, Camera *camera, DeferredRendererState &state,
                                       const std::vector<RenderObject> &renderObjects,
                                       const std::vector<glm::mat4> &models,
                                       const std::vector<Id> &transformIds)
{
    camera->clearColoring();

    // assign colors to render objects.
    uint32_t color = 1;

    int transformIdIndex = 0;
    for (size_t i = 0; i < renderObjects.size(); i++)
    {
        if (renderObjects[i].instanced)
        {
            for (size_t j = 0; j < renderObjects[i].instanceCount; j++)
            {
                unsigned char r = static_cast<unsigned char>(255 - ((color & 0x000000FF) >> 0));
                unsigned char g = static_cast<unsigned char>(255 - ((color & 0x0000FF00) >> 8));
                unsigned char b = static_cast<unsigned char>(255 - ((color & 0x00FF0000) >> 16));
                unsigned char a = static_cast<unsigned char>(255);

                camera->assignColoring(Color32(r, g, b, a), transformIds[transformIdIndex]);
                color++;
                transformIdIndex++;
            }
        }
        else
        {
            unsigned char r = static_cast<unsigned char>(255 - ((color & 0x000000FF) >> 0));
            unsigned char g = static_cast<unsigned char>(255 - ((color & 0x0000FF00) >> 8));
            unsigned char b = static_cast<unsigned char>(255 - ((color & 0x00FF0000) >> 16));
            unsigned char a = static_cast<unsigned char>(255);

            camera->assignColoring(Color32(r, g, b, a), transformIds[transformIdIndex]);
            color++;
            transformIdIndex++;
        }
    }

    //Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsColorPickingFBO());
    camera->getNativeGraphicsColorPickingFBO()->bind();
    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    color = 1;
    int modelIndex = 0;
    for (size_t i = 0; i < renderObjects.size(); i++)
    {
        if (renderObjects[i].instanced)
        {
            std::vector<glm::vec4> colors(renderObjects[i].instanceCount);
            for (size_t j = 0; j < renderObjects[i].instanceCount; j++)
            {
                unsigned char r = static_cast<unsigned char>(255 - ((color & 0x000000FF) >> 0));
                unsigned char g = static_cast<unsigned char>(255 - ((color & 0x0000FF00) >> 8));
                unsigned char b = static_cast<unsigned char>(255 - ((color & 0x00FF0000) >> 16));
                unsigned char a = static_cast<unsigned char>(255);

                colors[j] = glm::vec4(r, g, b, a);
                color++;
            }

            Renderer::getRenderer()->use(state.mColorInstancedShaderProgram);
            Renderer::getRenderer()->updateInstanceBuffer(renderObjects[i].instanceModelVbo, &models[modelIndex],
                                           renderObjects[i].instanceCount);
            Renderer::getRenderer()->updateInstanceColorBuffer(renderObjects[i].instanceColorVbo, &colors[0],
                                                renderObjects[i].instanceCount);
            Renderer::getRenderer()->renderInstanced(renderObjects[i], camera->mQuery);

            modelIndex += renderObjects[i].instanceCount;
        }
        else
        {
            unsigned char r = static_cast<unsigned char>(255 - ((color & 0x000000FF) >> 0));
            unsigned char g = static_cast<unsigned char>(255 - ((color & 0x0000FF00) >> 8));
            unsigned char b = static_cast<unsigned char>(255 - ((color & 0x00FF0000) >> 16));
            unsigned char a = static_cast<unsigned char>(255);

            color++;

            Renderer::getRenderer()->use(state.mColorShaderProgram);
            Renderer::getRenderer()->setMat4(state.mColorShaderModelLoc, models[modelIndex]);
            Renderer::getRenderer()->setColor32(state.mColorShaderColorLoc, Color32(r, g, b, a));

            Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);
            modelIndex++;
        }
    }

    //Renderer::getRenderer()->unbindFramebuffer();
    camera->getNativeGraphicsColorPickingFBO()->unbind();
}

static void endDeferredFrame(World *world, Camera *camera, DeferredRendererState &state)
{
    if (camera->mRenderToScreen)
    {
        Renderer::getRenderer()->bindFramebuffer(0);
        Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                              camera->getViewport().mHeight);

        Renderer::getRenderer()->use(state.mQuadShaderProgram);
        Renderer::getRenderer()->setTexture2D(state.mQuadShaderTexLoc, 0, *reinterpret_cast<unsigned int*>(camera->getNativeGraphicsColorTex()->getHandle()));

        Renderer::getRenderer()->renderScreenQuad(state.mQuadVAO);
        Renderer::getRenderer()->unbindFramebuffer();
    }

    camera->endQuery();
}