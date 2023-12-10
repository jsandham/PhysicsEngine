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

    mQuadShader = RendererShaders::getScreenQuadShader();
    mNormalsShader = RendererShaders::getNormalShader();
    mPositionShader = RendererShaders::getPositionShader();
    mLinearDepthShader = RendererShaders::getLinearDepthShader();
    mColorShader = RendererShaders::getColorShader();
    mNormalsInstancedShader = RendererShaders::getNormalInstancedShader();
    mPositionInstancedShader = RendererShaders::getPositionInstancedShader();
    mLinearDepthInstancedShader = RendererShaders::getLinearDepthInstancedShader();
    mColorInstancedShader = RendererShaders::getColorInstancedShader();

    mCameraUniform = RendererUniforms::getCameraUniform();

    mScreenQuad = RendererMeshes::getScreenQuad();

    Renderer::getRenderer()->turnOn(Capability::Depth_Testing);
}

void DebugRenderer::update(Camera *camera, const std::vector<DrawCallCommand> &commands,
                           const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds)
{
    beginDebugFrame(camera);

    renderDebug(camera, commands, models);

    renderDebugColorPicking(camera, commands, models, transformIds);

    endDebugFrame(camera);
}

void DebugRenderer::beginDebugFrame(Camera *camera)
{
    camera->beginQuery();

    mCameraUniform->setProjection(camera->getProjMatrix());
    mCameraUniform->setView(camera->getViewMatrix());
    mCameraUniform->setViewProjection(camera->getProjMatrix() * camera->getViewMatrix());
    mCameraUniform->setCameraPos(camera->getComponent<Transform>()->getPosition());

    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                         camera->getViewport().mWidth, camera->getViewport().mHeight);

    // update camera state data
    mCameraUniform->bind();
    mCameraUniform->copyToUniformsToDevice();

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

    camera->getNativeGraphicsColorPickingFBO()->bind();
    camera->getNativeGraphicsColorPickingFBO()->clearColor(Color::black);
    camera->getNativeGraphicsColorPickingFBO()->clearDepth(1.0f);
    camera->getNativeGraphicsColorPickingFBO()->unbind();
}

void DebugRenderer::renderDebug(Camera *camera, const std::vector<DrawCallCommand> &commands,
                                const std::vector<glm::mat4> &models)
{
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
    framebuffer->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                             camera->getViewport().mHeight);

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
            switch (camera->mColorTarget)
            {
            case ColorTarget::Normal:
                mNormalsInstancedShader->bind();
                break;
            case ColorTarget::Position:
                mPositionInstancedShader->bind();
                break;
            case ColorTarget::LinearDepth:
                mLinearDepthInstancedShader->bind();
                break;
            }

            VertexBuffer *instanceModelBuffer = mesh->getNativeGraphicsInstanceModelBuffer();

            instanceModelBuffer->bind(3);
            instanceModelBuffer->setData(models.data() + modelIndex, 0,
                                         sizeof(glm::mat4) * Renderer::INSTANCE_BATCH_SIZE);
            instanceModelBuffer->unbind(3);
            Renderer::getRenderer()->drawIndexedInstanced(mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
                                                          (subMeshVertexEndIndex - subMeshVertexStartIndex),
                                                          Renderer::INSTANCE_BATCH_SIZE,
                                                          camera->mQuery);
            modelIndex += Renderer::INSTANCE_BATCH_SIZE;
        }
        else
        {
            switch (camera->mColorTarget)
            {
            case ColorTarget::Normal:
                mNormalsShader->bind();
                mNormalsShader->setModel(models[modelIndex]);
                break;
            case ColorTarget::Position:
                mPositionShader->bind();
                mPositionShader->setModel(models[modelIndex]);
                break;
            case ColorTarget::LinearDepth:
                mLinearDepthShader->bind();
                mLinearDepthShader->setModel(models[modelIndex]);
                break;
            }

            Renderer::getRenderer()->drawIndexed(mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
                                                 (subMeshVertexEndIndex - subMeshVertexStartIndex), camera->mQuery);
            modelIndex++;
        }
    }

    framebuffer->unbind();
}

void DebugRenderer::renderDebugColorPicking(Camera *camera, const std::vector<DrawCallCommand> &commands,
                                            const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds)
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

            instanceModelBuffer->bind(3);
            instanceModelBuffer->setData(models.data() + modelIndex, 0,
                                         sizeof(glm::mat4) * Renderer::INSTANCE_BATCH_SIZE);
            instanceModelBuffer->unbind(3);

            instanceColorBuffer->bind(7);
            instanceColorBuffer->setData(colors.data(), 0, sizeof(glm::uvec4) * Renderer::INSTANCE_BATCH_SIZE);
            instanceColorBuffer->unbind(7);

            Renderer::getRenderer()->drawIndexedInstanced(mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
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

            Renderer::getRenderer()->drawIndexed(mesh->getNativeGraphicsHandle(),
                                                 subMeshVertexStartIndex,
                                                 (subMeshVertexEndIndex - subMeshVertexStartIndex),
                                                 camera->mQuery);

            color++;
            modelIndex++;
        }
    }

    camera->getNativeGraphicsColorPickingFBO()->unbind();
}

void DebugRenderer::endDebugFrame(Camera *camera)
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

    mCameraUniform->unbind();
}
