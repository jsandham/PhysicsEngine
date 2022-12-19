#include "../../include/graphics/DeferredRenderer.h"
#include "../../include/core/World.h"

#include "../../include/core/Log.h"

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

    initializeDeferredRenderer();
}

void DeferredRenderer::update(const Input &input, Camera *camera,
                              const std::vector<RenderObject> &renderObjects,
                              const std::vector<glm::mat4> &models,
                              const std::vector<Id> &transformIds)
{
    beginDeferredFrame(camera);

    geometryPass(camera, renderObjects, models);
    lightingPass(camera, renderObjects);

    renderColorPickingDeferred(camera, renderObjects, models, transformIds);

    endDeferredFrame(camera);
}

void DeferredRenderer::initializeDeferredRenderer()
{
    mState.mQuadShaderProgram = RendererShaders::getScreenQuadShader();
    mState.mGBufferShaderProgram = RendererShaders::getGBufferShader();
    mState.mColorShaderProgram = RendererShaders::getColorShader();
    mState.mColorInstancedShaderProgram = RendererShaders::getColorInstancedShader();

    mState.mQuadShaderTexLoc = mState.mQuadShaderProgram->findUniformLocation("screenTexture");
    mState.mGBufferShaderModelLoc = mState.mGBufferShaderProgram->findUniformLocation("model");
    mState.mGBufferShaderDiffuseTexLoc = mState.mGBufferShaderProgram->findUniformLocation("texture_diffuse1");
    mState.mGBufferShaderSpecTexLoc = mState.mGBufferShaderProgram->findUniformLocation("texture_specular1");
    mState.mColorShaderModelLoc = mState.mColorShaderProgram->findUniformLocation("model");
    mState.mColorShaderColorLoc = mState.mColorShaderProgram->findUniformLocation("material.color");

    Renderer::getRenderer()->createScreenQuad(&mState.mQuadVAO, &mState.mQuadVBO);

    Renderer::getRenderer()->createGlobalCameraUniforms(mState.mCameraState);

    Renderer::getRenderer()->turnOn(Capability::Depth_Testing);
}

void DeferredRenderer::beginDeferredFrame(Camera *camera)
{
    camera->beginQuery();

    mState.mCameraState.mProjection = camera->getProjMatrix();
    mState.mCameraState.mView = camera->getViewMatrix();
    mState.mCameraState.mViewProjection = camera->getProjMatrix() * camera->getViewMatrix();
    mState.mCameraState.mCameraPos = camera->getComponent<Transform>()->getPosition();

    // set camera state binding point and update camera state data
    Renderer::getRenderer()->setGlobalCameraUniforms(mState.mCameraState);

    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Framebuffer *framebuffer = nullptr;

    if (camera->mRenderTextureId.isValid())
    {
        RenderTexture *renderTexture = mWorld->getAssetByGuid<RenderTexture>(camera->mRenderTextureId);
        if (renderTexture != nullptr)
        {
            framebuffer = renderTexture->getNativeGraphicsMainFBO();
        }
        else
        {
            framebuffer = camera->getNativeGraphicsMainFBO();
        }
    }
    else
    {
        framebuffer = camera->getNativeGraphicsMainFBO();
    }

    framebuffer->bind();
    framebuffer->clearColor(camera->mBackgroundColor);
    framebuffer->clearDepth(1.0f);
    framebuffer->unbind();

    camera->getNativeGraphicsGeometryFBO()->bind();
    camera->getNativeGraphicsGeometryFBO()->clearColor(Color::black);
    camera->getNativeGraphicsGeometryFBO()->clearDepth(1.0f);
    camera->getNativeGraphicsGeometryFBO()->unbind();

    camera->getNativeGraphicsColorPickingFBO()->bind();
    camera->getNativeGraphicsColorPickingFBO()->clearColor(Color::black);
    camera->getNativeGraphicsColorPickingFBO()->clearDepth(1.0f);
    camera->getNativeGraphicsColorPickingFBO()->unbind();
}

void DeferredRenderer::geometryPass(Camera *camera, const std::vector<RenderObject> &renderObjects,
                                 const std::vector<glm::mat4> &models)
{
    // fill geometry framebuffer
    camera->getNativeGraphicsGeometryFBO()->bind();
    camera->getNativeGraphicsGeometryFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                        camera->getViewport().mWidth, camera->getViewport().mHeight);
    
    mState.mGBufferShaderProgram->bind();
    //Renderer::getRenderer()->use(state.mGBufferShaderProgram);
    //int mGBufferShaderDiffuseTexLoc = Renderer::getRenderer()->findUniformLocation("texture_diffuse1", state.mGBufferShaderProgram);
    //int mGBufferShaderSpecTexLoc = Renderer::getRenderer()->findUniformLocation("texture_specular1", state.mGBufferShaderProgram);

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
            Renderer::getRenderer()->setMat4(mState.mGBufferShaderModelLoc, models[modelIndex]);

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

    camera->getNativeGraphicsGeometryFBO()->unbind();
}

void DeferredRenderer::lightingPass(Camera *camera, const std::vector<RenderObject> &renderObjects)
{
    camera->getNativeGraphicsMainFBO()->bind();
    camera->getNativeGraphicsMainFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                    camera->getViewport().mWidth, camera->getViewport().mHeight);

    //Renderer::getRenderer()->use(state.mSimpleLitDeferredShaderProgram);

    //Renderer::getRenderer()->setTexture2D(state.mPositionTexLoc, 0, camera->getNativeGraphicsPositionTex());
    //Renderer::getRenderer()->setTexture2D(state.mNormalTexLoc, 1, camera->getNativeGraphicsNormalTex());
    //Renderer::getRenderer()->setTexture2D(state.mAlbedoSpecTexLoc, 2, camera->getNativeGraphicsAlbedoSpecTex());

    for (size_t i = 0; i < mWorld->getActiveScene()->getNumberOfComponents<Light>(); i++)
    {
        Light *light = mWorld->getActiveScene()->getComponentByIndex<Light>(i);
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

    camera->getNativeGraphicsMainFBO()->unbind();
}

void DeferredRenderer::renderColorPickingDeferred(Camera *camera,
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

    camera->getNativeGraphicsColorPickingFBO()->bind();
    camera->getNativeGraphicsColorPickingFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                            camera->getViewport().mWidth,
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

            mState.mColorInstancedShaderProgram->bind();
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

            mState.mColorShaderProgram->bind();
            mState.mColorShaderProgram->setMat4(mState.mColorShaderModelLoc, models[modelIndex]);
            mState.mColorShaderProgram->setColor32(mState.mColorShaderColorLoc, Color32(r, g, b, a));

            Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);
            modelIndex++;
        }
    }

    camera->getNativeGraphicsColorPickingFBO()->unbind();
}

void DeferredRenderer::endDeferredFrame(Camera *camera)
{
    if (camera->mRenderToScreen)
    {
        Renderer::getRenderer()->bindFramebuffer(0);
        Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                              camera->getViewport().mHeight);

        //Renderer::getRenderer()->use(state.mQuadShaderProgram);
        mState.mQuadShaderProgram->bind();
        Renderer::getRenderer()->setTexture2D(mState.mQuadShaderTexLoc, 0, camera->getNativeGraphicsColorTex());

        Renderer::getRenderer()->renderScreenQuad(mState.mQuadVAO);
        Renderer::getRenderer()->unbindFramebuffer();
    }

    camera->endQuery();
}