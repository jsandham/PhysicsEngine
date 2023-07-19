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

void DebugRenderer::update(const Input &input, Camera *camera, const std::vector<RenderObject> &renderObjects,
                           const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds)
{
    beginDebugFrame(camera);

    renderDebug(camera, renderObjects, models);

    renderDebugColorPicking(camera, renderObjects, models, transformIds);

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

void DebugRenderer::renderDebug(Camera *camera, const std::vector<RenderObject> &renderObjects,
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
    for (size_t i = 0; i < renderObjects.size(); i++)
    {
        if (renderObjects[i].instanced)
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

            renderObjects[i].instanceModelBuffer->bind();
            renderObjects[i].instanceModelBuffer->setData(models.data() + modelIndex, 0,
                                                          sizeof(glm::mat4) * renderObjects[i].instanceCount);
            renderObjects[i].instanceModelBuffer->unbind();
            Renderer::getRenderer()->drawIndexedInstanced(renderObjects[i], camera->mQuery);
            modelIndex += renderObjects[i].instanceCount;
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

            Renderer::getRenderer()->drawIndexed(renderObjects[i], camera->mQuery);
            modelIndex++;
        }
    }

    framebuffer->unbind();
}

void DebugRenderer::renderDebugColorPicking(Camera *camera, const std::vector<RenderObject> &renderObjects,
                                            const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds)
{
    camera->setColoringIds(transformIds);

    camera->getNativeGraphicsColorPickingFBO()->bind();
    camera->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                        camera->getViewport().mHeight);

    uint32_t color = 1;
    int modelIndex = 0;
    for (size_t i = 0; i < renderObjects.size(); i++)
    {
        if (renderObjects[i].instanced)
        {
            std::vector<glm::uvec4> colors(renderObjects[i].instanceCount);
            for (size_t j = 0; j < renderObjects[i].instanceCount; j++)
            {
                Color32 c = Color32::convertUint32ToColor32(color);
                colors[j].r = c.mR;
                colors[j].g = c.mG;
                colors[j].b = c.mB;
                colors[j].a = c.mA;
                color++;
            }

            mColorInstancedShader->bind();

            renderObjects[i].instanceModelBuffer->bind();
            renderObjects[i].instanceModelBuffer->setData(models.data() + modelIndex, 0,
                                                          sizeof(glm::mat4) * renderObjects[i].instanceCount);
            renderObjects[i].instanceModelBuffer->unbind();

            renderObjects[i].instanceColorBuffer->bind();
            renderObjects[i].instanceColorBuffer->setData(colors.data(), 0,
                                                          sizeof(glm::uvec4) * renderObjects[i].instanceCount);
            renderObjects[i].instanceColorBuffer->unbind();
            Renderer::getRenderer()->drawIndexedInstanced(renderObjects[i], camera->mQuery);

            modelIndex += renderObjects[i].instanceCount;
        }
        else
        {
            mColorShader->bind();
            mColorShader->setModel(models[modelIndex]);
            mColorShader->setColor32(Color32::convertUint32ToColor32(color));

            Renderer::getRenderer()->drawIndexed(renderObjects[i], camera->mQuery);

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
}
