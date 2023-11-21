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

    mScreenQuad = RendererMeshes::getScreenQuad();

    Renderer::getRenderer()->turnOn(Capability::Depth_Testing);
}

void DeferredRenderer::update(Camera *camera, const std::vector<DrawCallCommand> &commands,
                              const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds)
{
    beginDeferredFrame(camera);

    geometryPass(camera, commands, models);
    lightingPass(camera, commands);

    renderColorPickingDeferred(camera, commands, models, transformIds);

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

    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                         camera->getViewport().mWidth, camera->getViewport().mHeight);

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

void DeferredRenderer::geometryPass(Camera *camera, const std::vector<DrawCallCommand> &commands,
                                    const std::vector<glm::mat4> &models)
{
    // fill geometry framebuffer
    camera->getNativeGraphicsGeometryFBO()->bind();
    camera->getNativeGraphicsGeometryFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                        camera->getViewport().mWidth, camera->getViewport().mHeight);

    mGBufferShader->bind();
    // int mGBufferShaderDiffuseTexLoc = Renderer::getRenderer()->findUniformLocation("texture_diffuse1",
    // state.mGBufferShaderProgram); int mGBufferShaderSpecTexLoc =
    // Renderer::getRenderer()->findUniformLocation("texture_specular1", state.mGBufferShaderProgram);

    // std::string message = "mGBufferShaderDiffuseTexLoc: " + std::to_string(mGBufferShaderDiffuseTexLoc) +
    //                       " mGBufferShaderSpecTexLoc: " + std::to_string(mGBufferShaderSpecTexLoc) + "\n";
    // Log::info(message.c_str());

    // for (size_t i = 0; i < renderObjects.size(); i++)
    //{
    //     Renderer::getRenderer()->setMat4(state.mGBufferShaderModelLoc, renderObjects[i].model);
    //     //Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);
    //     Renderer::getRenderer()->draw(renderObjects[i], camera->mQuery);
    // }
    // int currentMaterialIndex = -1;
    // Material *material = nullptr;

    int modelIndex = 0;
    for (size_t i = 0; i < commands.size(); i++)
    {
        uint16_t meshIndex = commands[i].getMeshIndex();
        uint8_t subMesh = commands[i].getSubMesh();
        bool instanced = commands[i].isInstanced();

        Mesh *mesh = mWorld->getAssetByIndex<Mesh>(meshIndex);

        int subMeshVertexStartIndex = mesh->getSubMeshStartIndex(subMesh);
        int subMeshVertexEndIndex = mesh->getSubMeshEndIndex(subMesh);

        if (instanced)
        {
            modelIndex += Renderer::INSTANCE_BATCH_SIZE;
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

            Renderer::getRenderer()->drawIndexed(mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
                                                 (subMeshVertexEndIndex - subMeshVertexStartIndex),
                                                 camera->mQuery);
            modelIndex++;
        }
    }

    camera->getNativeGraphicsGeometryFBO()->unbind();
}

void DeferredRenderer::lightingPass(Camera *camera, const std::vector<DrawCallCommand> &commands)
{
    camera->getNativeGraphicsMainFBO()->bind();
    camera->getNativeGraphicsMainFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                    camera->getViewport().mWidth, camera->getViewport().mHeight);

    // Renderer::getRenderer()->use(state.mSimpleLitDeferredShaderProgram);

    // Renderer::getRenderer()->setTexture2D(state.mPositionTexLoc, 0, camera->getNativeGraphicsPositionTex());
    // Renderer::getRenderer()->setTexture2D(state.mNormalTexLoc, 1, camera->getNativeGraphicsNormalTex());
    // Renderer::getRenderer()->setTexture2D(state.mAlbedoSpecTexLoc, 2, camera->getNativeGraphicsAlbedoSpecTex());

    for (size_t i = 0; i < mWorld->getActiveScene()->getNumberOfComponents<Light>(); i++)
    {
        Light *light = mWorld->getActiveScene()->getComponentByIndex<Light>(i);
        Transform *lightTransform = light->getComponent<Transform>();
        if (lightTransform != nullptr)
        {
            // Renderer::getRenderer()->setVec3(state->mSimpleLitDeferredShaderLightPosLocs, lightTransform->mPosition);
            // Renderer::getRenderer()->setVec3(state->mSimpleLitDeferredShaderLightColLocs, light->mAmbient);
        }
    }

    // glBindVertexArray(state.mQuadVAO);
    // glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    // glBindVertexArray(0);

    camera->getNativeGraphicsMainFBO()->unbind();
}

void DeferredRenderer::renderColorPickingDeferred(Camera *camera, const std::vector<DrawCallCommand> &commands,
                                                  const std::vector<glm::mat4> &models,
                                                  const std::vector<Id> &transformIds)
{
    camera->setColoringIds(transformIds);

    camera->getNativeGraphicsColorPickingFBO()->bind();
    camera->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                        camera->getViewport().mHeight);

    uint32_t color = 1;
    int modelIndex = 0;
    for (size_t i = 0; i < commands.size(); i++)
    {
        uint16_t meshIndex = commands[i].getMeshIndex();
        uint8_t subMesh = commands[i].getSubMesh();
        bool instanced = commands[i].isInstanced();

        Mesh *mesh = mWorld->getAssetByIndex<Mesh>(meshIndex);

        int subMeshVertexStartIndex = mesh->getSubMeshStartIndex(subMesh);
        int subMeshVertexEndIndex = mesh->getSubMeshEndIndex(subMesh);

        if (instanced)
        {
            std::vector<glm::uvec4> colors(Renderer::INSTANCE_BATCH_SIZE);
            for (size_t j = 0; j < Renderer::INSTANCE_BATCH_SIZE; j++)
            {
                Color32 c = Color32::convertUint32ToColor32(color);
                colors[j].r = c.mR;
                colors[j].g = c.mG;
                colors[j].b = c.mB;
                colors[j].a = c.mA;
                color++;
            }

            VertexBuffer *instanceModelBuffer = mesh->getNativeGraphicsInstanceModelBuffer();
            VertexBuffer *instanceColorBuffer = mesh->getNativeGraphicsInstanceColorBuffer();

            mColorInstancedShader->bind();

            instanceModelBuffer->bind();
            instanceModelBuffer->setData(models.data() + modelIndex, 0,
                                         sizeof(glm::mat4) * Renderer::INSTANCE_BATCH_SIZE);
            instanceModelBuffer->unbind();

            instanceColorBuffer->bind();
            instanceColorBuffer->setData(colors.data(), 0, sizeof(glm::uvec4) * Renderer::INSTANCE_BATCH_SIZE);
            instanceColorBuffer->unbind();
            Renderer::getRenderer()->drawIndexedInstanced(
                mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
                                                          (subMeshVertexEndIndex - subMeshVertexStartIndex),
                                                          Renderer::INSTANCE_BATCH_SIZE,
                                                          camera->mQuery);

            modelIndex += Renderer::INSTANCE_BATCH_SIZE;
        }
        else
        {
            mColorShader->bind();
            mColorShader->setModel(models[modelIndex]);
            mColorShader->setColor32(Color32::convertUint32ToColor32(color));

            Renderer::getRenderer()->drawIndexed(mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
                                                 (subMeshVertexEndIndex - subMeshVertexStartIndex),
                                                 camera->mQuery);

            color++;
            modelIndex++;
        }
    }

    camera->getNativeGraphicsColorPickingFBO()->unbind();
}

void DeferredRenderer::endDeferredFrame(Camera *camera)
{
    if (camera->mRenderToScreen)
    {
        Renderer::getRenderer()->bindBackBuffer();
        Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                             camera->getViewport().mWidth, camera->getViewport().mHeight);

        mQuadShader->bind();
        mQuadShader->setScreenTexture(0, camera->getNativeGraphicsColorTex());

        mScreenQuad->draw();
        Renderer::getRenderer()->unbindBackBuffer();
    }

    camera->endQuery();
}