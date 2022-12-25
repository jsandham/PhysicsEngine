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

    mQuadShader = RendererShaders::getScreenQuadShader();
    mGBufferShader = RendererShaders::getGBufferShader();
    mColorShader = RendererShaders::getColorShader();
    mColorInstancedShader = RendererShaders::getColorInstancedShader();

    mCameraUniform = RendererUniforms::getCameraUniform();

    Renderer::getRenderer()->createScreenQuad(&mQuadVAO, &mQuadVBO);
    Renderer::getRenderer()->turnOn(Capability::Depth_Testing);
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

void DeferredRenderer::beginDeferredFrame(Camera *camera)
{
    camera->beginQuery();

    mCameraUniform->setProjection(camera->getProjMatrix());
    mCameraUniform->setView(camera->getViewMatrix());
    mCameraUniform->setViewProjection(camera->getProjMatrix() * camera->getViewMatrix());
    mCameraUniform->setCameraPos(camera->getComponent<Transform>()->getPosition());

    // set camera state binding point and update camera state data
    mCameraUniform->copyToUniformsToDevice();

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
    
    mGBufferShader->bind();
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
            mGBufferShader->setModel(models[modelIndex]);

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

            mColorInstancedShader->bind();
           
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

            mColorShader->bind();
            mColorShader->setModel(models[modelIndex]);
            mColorShader->setColor(Color32(r, g, b, a));

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

        mQuadShader->bind();
        mQuadShader->setScreenTexture(0, camera->getNativeGraphicsColorTex());

        Renderer::getRenderer()->renderScreenQuad(mQuadVAO);
        Renderer::getRenderer()->unbindFramebuffer();
    }

    camera->endQuery();
}