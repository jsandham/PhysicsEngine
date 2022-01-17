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

    initializeDeferredRenderer(mWorld, mState);
}

void DeferredRenderer::update(const Input &input, Camera *camera,
                              const std::vector<RenderObject> &renderObjects,
                              const std::vector<glm::mat4> &models,
                              const std::vector<Guid> &transformIds)
{
    beginDeferredFrame(mWorld, camera, mState);

    geometryPass(mWorld, camera, mState, renderObjects, models);
    lightingPass(mWorld, camera, mState, renderObjects);

    renderColorPickingDeferred(mWorld, camera, mState, renderObjects, models, transformIds);

    endDeferredFrame(mWorld, camera, mState);
}

void PhysicsEngine::initializeDeferredRenderer(World *world, DeferredRendererState &state)
{
    Graphics::compileScreenQuadShader(state);
    Graphics::compileGBufferShader(state);
    Graphics::compileColorShader(state);
    Graphics::compileColorInstancedShader(state);

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
    state.mCameraState.mViewProjection = camera->getProjMatrix() * camera->getViewMatrix();
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
                                 const std::vector<RenderObject> &renderObjects,
                                 const std::vector<glm::mat4> &models)
{
    // fill geometry framebuffer
    Graphics::bindFramebuffer(camera->getNativeGraphicsGeometryFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);
    
    Graphics::use(state.mGBufferShaderProgram);

    int mGBufferShaderDiffuseTexLoc = Graphics::findUniformLocation("texture_diffuse1", state.mGBufferShaderProgram);
    int mGBufferShaderSpecTexLoc = Graphics::findUniformLocation("texture_specular1", state.mGBufferShaderProgram);

    //std::string message = "mGBufferShaderDiffuseTexLoc: " + std::to_string(mGBufferShaderDiffuseTexLoc) +
    //                      " mGBufferShaderSpecTexLoc: " + std::to_string(mGBufferShaderSpecTexLoc) + "\n";
    //Log::info(message.c_str());

    //for (size_t i = 0; i < renderObjects.size(); i++)
    //{
    //    Graphics::setMat4(state.mGBufferShaderModelLoc, renderObjects[i].model);
    //    Graphics::render(renderObjects[i], camera->mQuery);
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
            Graphics::setMat4(state.mGBufferShaderModelLoc, models[modelIndex]);

            // if (currentMaterialIndex != renderObjects[renderQueue[i].second].materialIndex)
            //{
            //    material = world->getAssetByIndex<Material>(renderObjects[renderQueue[i].second].materialIndex);
            //    material->apply(world);
            //
            //    currentMaterialIndex = renderObjects[renderQueue[i].second].materialIndex;
            //}

            Graphics::render(renderObjects[i], camera->mQuery);
            modelIndex++;
        }
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
                                       const std::vector<RenderObject> &renderObjects,
                                       const std::vector<glm::mat4> &models,
                                       const std::vector<Guid> &transformIds)
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

    Graphics::bindFramebuffer(camera->getNativeGraphicsColorPickingFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
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

            Graphics::use(state.mColorInstancedShaderProgram);
            Graphics::updateInstanceBuffer(renderObjects[i].vbo, &models[modelIndex], renderObjects[i].instanceCount);
            Graphics::updateInstanceColorBuffer(renderObjects[i].vbo2, &colors[0], renderObjects[i].instanceCount);
            Graphics::renderInstanced(renderObjects[i], camera->mQuery);

            modelIndex += renderObjects[i].instanceCount;
        }
        else
        {
            unsigned char r = static_cast<unsigned char>(255 - ((color & 0x000000FF) >> 0));
            unsigned char g = static_cast<unsigned char>(255 - ((color & 0x0000FF00) >> 8));
            unsigned char b = static_cast<unsigned char>(255 - ((color & 0x00FF0000) >> 16));
            unsigned char a = static_cast<unsigned char>(255);

            color++;

            Graphics::use(state.mColorShaderProgram);
            Graphics::setMat4(state.mColorShaderModelLoc, models[modelIndex]);
            Graphics::setColor32(state.mColorShaderColorLoc, Color32(r, g, b, a));

            Graphics::render(renderObjects[i], camera->mQuery);
            modelIndex++;
        }
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