#include "../../include/graphics/DebugRenderer.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

static void initializeDebugRenderer(World* world, DebugRendererState& state);
static void beginDebugFrame(World* world, Camera* camera, DebugRendererState& state);
static void renderDebug(World* world, Camera* camera, DebugRendererState& state,
    const std::vector<RenderObject>& renderObjects,
    const std::vector<glm::mat4>& models);
static void renderDebugColorPicking(World* world, Camera* camera, DebugRendererState& state,
    const std::vector<RenderObject>& renderObjects,
    const std::vector<glm::mat4>& models,
    const std::vector<Id>& transformIds);
static void endDebugFrame(World* world, Camera* camera, DebugRendererState& state);

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
    const std::vector<Id> &transformIds)
{
    beginDebugFrame(mWorld, camera, mState);

    renderDebug(mWorld, camera, mState, renderObjects, models);

    renderDebugColorPicking(mWorld, camera, mState, renderObjects, models, transformIds);

    endDebugFrame(mWorld, camera, mState);
}

static void initializeDebugRenderer(World *world, DebugRendererState &state)
{
    state.mQuadShaderProgram = RendererShaders::getScreenQuadShader();
    state.mNormalsShaderProgram = RendererShaders::getNormalShader();
    state.mPositionShaderProgram = RendererShaders::getPositionShader();
    state.mLinearDepthShaderProgram = RendererShaders::getLinearDepthShader();
    state.mColorShaderProgram = RendererShaders::getColorShader();
    state.mNormalsInstancedShaderProgram = RendererShaders::getNormalInstancedShader();
    state.mPositionInstancedShaderProgram = RendererShaders::getPositionInstancedShader();
    state.mLinearDepthInstancedShaderProgram = RendererShaders::getLinearDepthInstancedShader();
    state.mColorInstancedShaderProgram = RendererShaders::getColorInstancedShader();

    state.mQuadShaderTexLoc = state.mQuadShaderProgram->findUniformLocation("screenTexture");
    state.mNormalsShaderModelLoc = state.mNormalsShaderProgram->findUniformLocation("model");
    state.mPositionShaderModelLoc = state.mPositionShaderProgram->findUniformLocation("model");
    state.mLinearDepthShaderModelLoc = state.mLinearDepthShaderProgram->findUniformLocation("model");
    state.mColorShaderModelLoc = state.mColorShaderProgram->findUniformLocation("model");
    state.mColorShaderModelLoc = state.mColorShaderProgram->findUniformLocation("material.color");


    /*state.mNormalsShaderProgram = RendererShaders::getRendererShaders()->getNormalShader().mProgram;
    state.mNormalsShaderModelLoc = RendererShaders::getRendererShaders()->getNormalShader().mModelLoc;
    state.mNormalsInstancedShaderProgram = RendererShaders::getRendererShaders()->getNormalInstancedShader().mProgram;

    state.mPositionShaderProgram = RendererShaders::getRendererShaders()->getPositionShader().mProgram;
    state.mPositionShaderModelLoc = RendererShaders::getRendererShaders()->getPositionShader().mModelLoc;
    state.mPositionInstancedShaderProgram = RendererShaders::getRendererShaders()->getPositionInstancedShader().mProgram;

    state.mLinearDepthShaderProgram = RendererShaders::getRendererShaders()->getLinearDepthShader().mProgram;
    state.mLinearDepthShaderModelLoc = RendererShaders::getRendererShaders()->getLinearDepthShader().mModelLoc;
    state.mLinearDepthInstancedShaderProgram = RendererShaders::getRendererShaders()->getLinearDepthInstancedShader().mProgram;

    state.mColorShaderProgram = RendererShaders::getRendererShaders()->getColorShader().mProgram;
    state.mColorShaderModelLoc = RendererShaders::getRendererShaders()->getColorShader().mModelLoc;
    state.mColorShaderColorLoc = RendererShaders::getRendererShaders()->getColorShader().mColorLoc;
    state.mColorInstancedShaderProgram = RendererShaders::getRendererShaders()->getColorInstancedShader().mProgram;

    state.mQuadShaderProgram = RendererShaders::getRendererShaders()->getScreenQuadShader().mProgram;
    state.mQuadShaderTexLoc = RendererShaders::getRendererShaders()->getScreenQuadShader().mTexLoc;*/

    Renderer::getRenderer()->createScreenQuad(&state.mQuadVAO, &state.mQuadVBO);

    Renderer::getRenderer()->createGlobalCameraUniforms(state.mCameraState);

    Renderer::getRenderer()->turnOn(Capability::Depth_Testing);
}

static void beginDebugFrame(World *world, Camera *camera, DebugRendererState &state)
{
    camera->beginQuery();

    state.mCameraState.mProjection = camera->getProjMatrix();
    state.mCameraState.mView = camera->getViewMatrix();
    state.mCameraState.mViewProjection = camera->getProjMatrix() * camera->getViewMatrix();
    state.mCameraState.mCameraPos = camera->getComponent<Transform>()->getPosition();

    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    // update camera state data
    Renderer::getRenderer()->setGlobalCameraUniforms(state.mCameraState);

    Framebuffer* framebuffer = nullptr;

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

    // Renderer::getRenderer()->clearFrambufferColor(camera->mBackgroundColor);
    // Renderer::getRenderer()->clearFramebufferDepth(1.0f);
    // Renderer::getRenderer()->unbindFramebuffer();
    framebuffer->bind();
    framebuffer->clearColor(camera->mBackgroundColor);
    framebuffer->clearDepth(1.0f);
    framebuffer->unbind();

    //Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsColorPickingFBO());
    //Renderer::getRenderer()->clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
    //Renderer::getRenderer()->clearFramebufferDepth(1.0f);
    //Renderer::getRenderer()->unbindFramebuffer();
    camera->getNativeGraphicsColorPickingFBO()->bind();
    camera->getNativeGraphicsColorPickingFBO()->clearColor(Color::black);
    camera->getNativeGraphicsColorPickingFBO()->clearDepth(1.0f);
    camera->getNativeGraphicsColorPickingFBO()->unbind();
}

static void renderDebug(World* world, Camera* camera, DebugRendererState& state,
    const std::vector<RenderObject>& renderObjects, const std::vector<glm::mat4> &models)
{
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

    framebuffer->bind();

    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    int modelIndex = 0;
    for (size_t i = 0; i < renderObjects.size(); i++)
    {
        if (renderObjects[i].instanced)
        {
            switch (camera->mColorTarget)
            {
            case ColorTarget::Normal:
                state.mNormalsInstancedShaderProgram->bind();
                //Renderer::getRenderer()->use(state.mNormalsInstancedShaderProgram);
                break;
            case ColorTarget::Position:
                state.mPositionInstancedShaderProgram->bind();
                //Renderer::getRenderer()->use(state.mPositionInstancedShaderProgram);
                break;
            case ColorTarget::LinearDepth:
                state.mLinearDepthInstancedShaderProgram->bind();
                //Renderer::getRenderer()->use(state.mLinearDepthInstancedShaderProgram);
                break;
            }

            Renderer::getRenderer()->updateInstanceBuffer(renderObjects[i].instanceModelVbo, &models[modelIndex],
                                           renderObjects[i].instanceCount);
            Renderer::getRenderer()->renderInstanced(renderObjects[i], camera->mQuery);
            modelIndex += renderObjects[i].instanceCount;
        }
        else
        {
            int modelLoc = -1;
            switch (camera->mColorTarget)
            {
            case ColorTarget::Normal:
                state.mNormalsShaderProgram->bind();
                //Renderer::getRenderer()->use(state.mNormalsShaderProgram);
                modelLoc = state.mNormalsShaderModelLoc;
                break;
            case ColorTarget::Position:
                state.mPositionShaderProgram->bind();
                //Renderer::getRenderer()->use(state.mPositionShaderProgram);
                modelLoc = state.mPositionShaderModelLoc;
                break;
            case ColorTarget::LinearDepth:
                state.mLinearDepthShaderProgram->bind();
                //Renderer::getRenderer()->use(state.mLinearDepthShaderProgram);
                modelLoc = state.mLinearDepthShaderModelLoc;
                break;
            }

            assert(modelLoc != -1);

            Renderer::getRenderer()->setMat4(modelLoc, models[modelIndex]);
            Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);
            modelIndex++;
        }
    }

    //Renderer::getRenderer()->unbindFramebuffer();
    framebuffer->unbind();
}

static void renderDebugColorPicking(World *world, Camera *camera, DebugRendererState &state,
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

            state.mColorInstancedShaderProgram->bind();
            //Renderer::getRenderer()->use(state.mColorInstancedShaderProgram);
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

            //Renderer::getRenderer()->use(state.mColorShaderProgram);
            state.mColorShaderProgram->bind();
            Renderer::getRenderer()->setMat4(state.mColorShaderModelLoc, models[modelIndex]);
            Renderer::getRenderer()->setColor32(state.mColorShaderColorLoc, Color32(r, g, b, a));

            Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);
            modelIndex++;
        }
    }

    //Renderer::getRenderer()->unbindFramebuffer();
    camera->getNativeGraphicsColorPickingFBO()->unbind();
}

static void endDebugFrame(World *world, Camera *camera, DebugRendererState &state)
{
    if (camera->mRenderToScreen)
    {
        Renderer::getRenderer()->bindFramebuffer(0);
        Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                              camera->getViewport().mHeight);

        //Renderer::getRenderer()->use(state.mQuadShaderProgram);
        state.mQuadShaderProgram->bind();
        Renderer::getRenderer()->setTexture2D(state.mQuadShaderTexLoc, 0, camera->getNativeGraphicsColorTex());

        Renderer::getRenderer()->renderScreenQuad(state.mQuadVAO);
        Renderer::getRenderer()->unbindFramebuffer();
    }

    camera->endQuery();
}
