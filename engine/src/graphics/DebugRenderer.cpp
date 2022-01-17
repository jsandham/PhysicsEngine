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
    const std::vector<RenderObject>& renderObjects,
    const std::vector<glm::mat4> &models,
    const std::vector<Guid> &transformIds)
{
    beginDebugFrame(mWorld, camera, mState);

    renderDebug(mWorld, camera, mState, renderObjects, models);

    renderDebugColorPicking(mWorld, camera, mState, renderObjects, models, transformIds);

    endDebugFrame(mWorld, camera, mState);
}

void PhysicsEngine::initializeDebugRenderer(World *world, DebugRendererState &state)
{
    Graphics::compileNormalShader(state);
    Graphics::compileNormalInstancedShader(state);
    Graphics::compilePositionShader(state);
    Graphics::compilePositionInstancedShader(state);
    Graphics::compileLinearDepthShader(state);
    Graphics::compileLinearDepthInstancedShader(state);
    Graphics::compileColorShader(state);
    Graphics::compileColorInstancedShader(state);
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
    state.mCameraState.mViewProjection = camera->getProjMatrix() * camera->getViewMatrix();
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
    const std::vector<RenderObject>& renderObjects, const std::vector<glm::mat4> &models)
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

    int modelIndex = 0;
    for (size_t i = 0; i < renderObjects.size(); i++)
    {
        if (renderObjects[i].instanced)
        {
            switch (camera->mColorTarget)
            {
            case ColorTarget::Normal:
                Graphics::use(state.mNormalsInstancedShaderProgram);
                break;
            case ColorTarget::Position:
                Graphics::use(state.mPositionInstancedShaderProgram);
                break;
            case ColorTarget::LinearDepth:
                Graphics::use(state.mLinearDepthInstancedShaderProgram);
                break;
            }

            Graphics::updateInstanceBuffer(renderObjects[i].vbo, &models[modelIndex], renderObjects[i].instanceCount);
            Graphics::renderInstanced(renderObjects[i], camera->mQuery);
            modelIndex += renderObjects[i].instanceCount;
        }
        else
        {
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

            Graphics::setMat4(modelLoc, models[modelIndex]);
            Graphics::render(renderObjects[i], camera->mQuery);
            modelIndex++;
        }
    }

    Graphics::unbindFramebuffer();
}

void PhysicsEngine::renderDebugColorPicking(World *world, Camera *camera, DebugRendererState &state,
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
