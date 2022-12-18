#include "../../include/graphics/ForwardRenderer.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

static void initializeRenderer(World* world, ForwardRendererState& state);
static void beginFrame(World* world, Camera* camera, ForwardRendererState& state);
static void computeSSAO(World* world, Camera* camera, ForwardRendererState& state,
    const std::vector<RenderObject>& renderObjects, const std::vector<glm::mat4>& models);
static void renderShadows(World* world, Camera* camera, Light* light, Transform* lightTransform,
    ForwardRendererState& state, const std::vector<RenderObject>& renderObjects,
    const std::vector<glm::mat4>& models);
static void renderOpaques(World* world, Camera* camera, Light* light, Transform* lightTransform,
    ForwardRendererState& state, const std::vector<RenderObject>& renderObjects,
    const std::vector<glm::mat4>& models);
static void renderSprites(World* world, Camera* camera, ForwardRendererState& state,
    const std::vector<SpriteObject>& spriteObjects);
static void renderColorPicking(World* world, Camera* camera, ForwardRendererState& state,
    const std::vector<RenderObject>& renderObjects,
    const std::vector<glm::mat4>& models, const std::vector<Id>& transformIds);
static void renderTransparents();
static void postProcessing();
static void endFrame(World* world, Camera* camera, ForwardRendererState& state);
static void calcCascadeOrthoProj(Camera* camera, glm::vec3 lightDirection, ForwardRendererState& state);

ForwardRenderer::ForwardRenderer()
{
}

ForwardRenderer::~ForwardRenderer()
{
}

void ForwardRenderer::init(World *world)
{
    mWorld = world;

    initializeRenderer(mWorld, mState);
}

void ForwardRenderer::update(const Input &input, Camera *camera,
                             const std::vector<RenderObject> &renderObjects,
                             const std::vector<glm::mat4> &models,
                             const std::vector<Id> &transformIds,
                             const std::vector<SpriteObject> &spriteObjects)
{
    beginFrame(mWorld, camera, mState);

    if (camera->mSSAO == CameraSSAO::SSAO_On)
    {
        computeSSAO(mWorld, camera, mState, renderObjects, models);
    }

    for (size_t j = 0; j < mWorld->getActiveScene()->getNumberOfComponents<Light>(); j++)
    {
        Light *light = mWorld->getActiveScene()->getComponentByIndex<Light>(j);
        
        if (light->mEnabled)
        {
            Transform* lightTransform = light->getComponent<Transform>();

            if (lightTransform != nullptr)
            {
                if (light->mShadowType != ShadowType::None)
                {
                    renderShadows(mWorld, camera, light, lightTransform, mState, renderObjects, models);
                }
                
                renderOpaques(mWorld, camera, light, lightTransform, mState, renderObjects, models);
                
                renderTransparents();
            }
        }
    }

    renderSprites(mWorld, camera, mState, spriteObjects);

    renderColorPicking(mWorld, camera, mState, renderObjects, models, transformIds);

    postProcessing();

    endFrame(mWorld, camera, mState);
}

static void initializeRenderer(World *world, ForwardRendererState &state)
{
    state.mQuadShaderProgram = RendererShaders::getScreenQuadShader();
    state.mDepthShaderProgram = RendererShaders::getDepthShader();
    state.mDepthCubemapShaderProgram = RendererShaders::getDepthCubemapShader();
    state.mGeometryShaderProgram = RendererShaders::getGeometryShader();
    state.mColorShaderProgram = RendererShaders::getColorShader();
    state.mColorInstancedShaderProgram = RendererShaders::getColorInstancedShader();
    state.mSsaoShaderProgram = RendererShaders::getSSAOShader();
    state.mSpriteShaderProgram = RendererShaders::getSpriteShader();

    state.mQuadShaderTexLoc = state.mQuadShaderProgram->findUniformLocation("screenTexture");
    state.mDepthShaderModelLoc = state.mDepthShaderProgram->findUniformLocation("model");
    state.mDepthShaderViewLoc = state.mDepthShaderProgram->findUniformLocation("view");
    state.mDepthShaderProjectionLoc = state.mDepthShaderProgram->findUniformLocation("projection");
    state.mDepthCubemapShaderLightPosLoc = state.mDepthCubemapShaderProgram->findUniformLocation("lightPos");
    state.mDepthCubemapShaderFarPlaneLoc = state.mDepthCubemapShaderProgram->findUniformLocation("farPlane");
    state.mDepthCubemapShaderModelLoc = state.mDepthCubemapShaderProgram->findUniformLocation("model");
    state.mDepthCubemapShaderCubeViewProjMatricesLoc0 =
        state.mDepthCubemapShaderProgram->findUniformLocation("cubeViewProjMatrices[0]");
    state.mDepthCubemapShaderCubeViewProjMatricesLoc1 =
        state.mDepthCubemapShaderProgram->findUniformLocation("cubeViewProjMatrices[1]");
    state.mDepthCubemapShaderCubeViewProjMatricesLoc2 =
        state.mDepthCubemapShaderProgram->findUniformLocation("cubeViewProjMatrices[2]");
    state.mDepthCubemapShaderCubeViewProjMatricesLoc3 =
        state.mDepthCubemapShaderProgram->findUniformLocation("cubeViewProjMatrices[3]");
    state.mDepthCubemapShaderCubeViewProjMatricesLoc4 =
        state.mDepthCubemapShaderProgram->findUniformLocation("cubeViewProjMatrices[4]");
    state.mDepthCubemapShaderCubeViewProjMatricesLoc5 =
        state.mDepthCubemapShaderProgram->findUniformLocation("cubeViewProjMatrices[5]");
    state.mGeometryShaderModelLoc = state.mGeometryShaderProgram->findUniformLocation("model");
    state.mColorShaderModelLoc = state.mColorShaderProgram->findUniformLocation("model");
    state.mColorShaderColorLoc = state.mColorShaderProgram->findUniformLocation("material.color");
    state.mSsaoShaderProjectionLoc = state.mSsaoShaderProgram->findUniformLocation("projection");
    state.mSsaoShaderPositionTexLoc = state.mSsaoShaderProgram->findUniformLocation("positionTex");
    state.mSsaoShaderNormalTexLoc = state.mSsaoShaderProgram->findUniformLocation("normalTex");
    state.mSsaoShaderNoiseTexLoc = state.mSsaoShaderProgram->findUniformLocation("noiseTex");
    for (int i = 0; i < 64; i++)
    {
        state.mSsaoShaderSamplesLoc[i] =
            state.mSsaoShaderProgram->findUniformLocation("samples[" + std::to_string(i) + "]");
    }
    state.mSpriteModelLoc = state.mSpriteShaderProgram->findUniformLocation("model");
    state.mSpriteViewLoc = state.mSpriteShaderProgram->findUniformLocation("view");
    state.mSpriteProjectionLoc = state.mSpriteShaderProgram->findUniformLocation("projection");
    state.mSpriteColorLoc = state.mSpriteShaderProgram->findUniformLocation("spriteColor");
    state.mSpriteImageLoc = state.mSpriteShaderProgram->findUniformLocation("image");

    Renderer::getRenderer()->createScreenQuad(&state.mQuadVAO, &state.mQuadVBO);

    Renderer::getRenderer()->createGlobalCameraUniforms(state.mCameraState);
    Renderer::getRenderer()->createGlobalLightUniforms(state.mLightState);

    Renderer::getRenderer()->turnOn(Capability::Depth_Testing);
}

static void beginFrame(World *world, Camera *camera, ForwardRendererState &state)
{
    Renderer::getRenderer()->turnOn(Capability::BackfaceCulling);

    camera->beginQuery();

    state.mCameraState.mProjection = camera->getProjMatrix();
    state.mCameraState.mView = camera->getViewMatrix();
    state.mCameraState.mViewProjection = camera->getProjMatrix() * camera->getViewMatrix();
    state.mCameraState.mCameraPos = camera->getComponent<Transform>()->getPosition();

    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    // update camera state data
    Renderer::getRenderer()->setGlobalCameraUniforms(state.mCameraState);

    Framebuffer *framebuffer = nullptr;

    if (camera->mRenderTextureId.isValid())
    {
        RenderTexture *renderTexture = world->getAssetByGuid<RenderTexture>(camera->mRenderTextureId);
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

    if(camera->mSSAO == CameraSSAO::SSAO_On)
    {
        camera->getNativeGraphicsSSAOFBO()->bind();
        camera->getNativeGraphicsSSAOFBO()->clearColor(Color::black);
        camera->getNativeGraphicsSSAOFBO()->unbind();
    }
}

static void computeSSAO(World *world, Camera *camera, ForwardRendererState &state,
                                const std::vector<RenderObject> &renderObjects, const std::vector<glm::mat4> &models)
{
    // fill geometry framebuffer
    camera->getNativeGraphicsGeometryFBO()->bind();
    camera->getNativeGraphicsGeometryFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                        camera->getViewport().mWidth, camera->getViewport().mHeight);

    state.mGeometryShaderProgram->bind();

    int modelIndex = 0;
    for (size_t i = 0; i < renderObjects.size(); i++)
    {
        if (renderObjects[i].instanced)
        {
            modelIndex += renderObjects[i].instanceCount;
        }
        else
        {
            state.mGeometryShaderProgram->setMat4(state.mGeometryShaderModelLoc, models[modelIndex]);
            Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);
            modelIndex++;
        }
    }

    camera->getNativeGraphicsGeometryFBO()->unbind();

    // fill ssao color texture
    camera->getNativeGraphicsSSAOFBO()->bind();
    camera->getNativeGraphicsSSAOFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                    camera->getViewport().mWidth, camera->getViewport().mHeight);

    state.mSsaoShaderProgram->bind();
    state.mSsaoShaderProgram->setMat4(state.mSsaoShaderProjectionLoc, camera->getProjMatrix());
    for (int i = 0; i < 64; i++)
    {
        state.mSsaoShaderProgram->setVec3(state.mSsaoShaderSamplesLoc[i], camera->getSSAOSample(i));
    }
    state.mSsaoShaderProgram->setTexture2D(state.mSsaoShaderPositionTexLoc, 0, camera->getNativeGraphicsPositionTex());
    state.mSsaoShaderProgram->setTexture2D(state.mSsaoShaderNormalTexLoc, 1, camera->getNativeGraphicsNormalTex());
    state.mSsaoShaderProgram->setTexture2D(state.mSsaoShaderNoiseTexLoc, 2, camera->getNativeGraphicsSSAONoiseTex());

    Renderer::getRenderer()->renderScreenQuad(state.mQuadVAO);

    camera->getNativeGraphicsSSAOFBO()->unbind();
}

static void renderShadows(World *world, Camera *camera, Light *light, Transform *lightTransform,
                                  ForwardRendererState &state, const std::vector<RenderObject> &renderObjects, 
                                  const std::vector<glm::mat4> &models)
{
    if (light->mLightType == LightType::Directional)
    {
        std::array<float, 6> cascadeEnds = camera->calcViewSpaceCascadeEnds();
        for (size_t i = 0; i < cascadeEnds.size(); i++)
        {
            state.mCascadeEnds[i] = cascadeEnds[i];
        }

        calcCascadeOrthoProj(camera, lightTransform->getForward(), state);

        for (int i = 0; i < 5; i++)
        {
            light->getNativeGraphicsShadowCascadeFBO(i)->bind();
            light->getNativeGraphicsShadowCascadeFBO(i)->setViewport(0, 0,
                                                                     static_cast<int>(light->getShadowMapResolution()),
                                                                     static_cast<int>(light->getShadowMapResolution()));
            light->getNativeGraphicsShadowCascadeFBO(i)->clearDepth(1.0f);

            state.mDepthShaderProgram->bind();
            state.mDepthShaderProgram->setMat4(state.mDepthShaderViewLoc, state.mCascadeLightView[i]);
            state.mDepthShaderProgram->setMat4(state.mDepthShaderProjectionLoc, state.mCascadeOrthoProj[i]);

            int modelIndex = 0;
            for (size_t j = 0; j < renderObjects.size(); j++)
            {
                if (renderObjects[j].instanced)
                {
                    modelIndex += renderObjects[j].instanceCount;                    
                }
                else
                {
                    state.mDepthShaderProgram->setMat4(state.mDepthShaderModelLoc, models[modelIndex]);
                    Renderer::getRenderer()->render(renderObjects[j], camera->mQuery);
                    modelIndex++;
                }
            }

            light->getNativeGraphicsShadowCascadeFBO(i)->unbind();
        }
    }
    else if (light->mLightType == LightType::Spot)
    {
        light->getNativeGraphicsShadowSpotlightFBO()->bind();
        light->getNativeGraphicsShadowSpotlightFBO()->setViewport(
            0, 0, static_cast<int>(light->getShadowMapResolution()), static_cast<int>(light->getShadowMapResolution()));
        light->getNativeGraphicsShadowSpotlightFBO()->clearDepth(1.0f);

        state.mShadowProjMatrix = light->getProjMatrix();
        state.mShadowViewMatrix =
            glm::lookAt(lightTransform->getPosition(), lightTransform->getPosition() + lightTransform->getForward(),
                        glm::vec3(0.0f, 1.0f, 0.0f));

        state.mDepthShaderProgram->bind();
        state.mDepthShaderProgram->setMat4(state.mDepthShaderProjectionLoc, state.mShadowProjMatrix);
        state.mDepthShaderProgram->setMat4(state.mDepthShaderViewLoc, state.mShadowViewMatrix);

        int modelIndex = 0;
        for (size_t i = 0; i < renderObjects.size(); i++)
        {
            if (renderObjects[i].instanced)
            {
                modelIndex += renderObjects[i].instanceCount;
            }
            else
            {
                state.mDepthShaderProgram->setMat4(state.mDepthShaderModelLoc, models[modelIndex]);
                
                Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);
                modelIndex++;
            }
        }

        light->getNativeGraphicsShadowSpotlightFBO()->unbind();
    }
    else if (light->mLightType == LightType::Point)
    {

        state.mCubeViewProjMatrices[0] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->getPosition(),
                                                  lightTransform->getPosition() + glm::vec3(1.0, 0.0, 0.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));
        state.mCubeViewProjMatrices[1] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->getPosition(),
                                                  lightTransform->getPosition() + glm::vec3(-1.0, 0.0, 0.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));
        state.mCubeViewProjMatrices[2] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->getPosition(),
                                                  lightTransform->getPosition() + glm::vec3(0.0, 1.0, 0.0),
                                                  glm::vec3(0.0, 0.0, 1.0)));
        state.mCubeViewProjMatrices[3] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->getPosition(),
                                                  lightTransform->getPosition() + glm::vec3(0.0, -1.0, 0.0),
                                                  glm::vec3(0.0, 0.0, -1.0)));
        state.mCubeViewProjMatrices[4] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->getPosition(),
                                                  lightTransform->getPosition() + glm::vec3(0.0, 0.0, 1.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));
        state.mCubeViewProjMatrices[5] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->getPosition(),
                                                  lightTransform->getPosition() + glm::vec3(0.0, 0.0, -1.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));

        light->getNativeGraphicsShadowCubemapFBO()->bind();
        light->getNativeGraphicsShadowCubemapFBO()->setViewport(0, 0, static_cast<int>(light->getShadowMapResolution()),
                                                                static_cast<int>(light->getShadowMapResolution()));
        light->getNativeGraphicsShadowCubemapFBO()->clearDepth(1.0f);

        state.mDepthCubemapShaderProgram->bind();
        state.mDepthCubemapShaderProgram->setVec3(state.mDepthCubemapShaderLightPosLoc, lightTransform->getPosition());
        state.mDepthCubemapShaderProgram->setFloat(state.mDepthCubemapShaderFarPlaneLoc, camera->getFrustum().mFarPlane);
        state.mDepthCubemapShaderProgram->setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc0,
                                                  state.mCubeViewProjMatrices[0]);
        state.mDepthCubemapShaderProgram->setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc1,
                                                  state.mCubeViewProjMatrices[1]);
        state.mDepthCubemapShaderProgram->setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc2,
                                                  state.mCubeViewProjMatrices[2]);
        state.mDepthCubemapShaderProgram->setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc3,
                                                  state.mCubeViewProjMatrices[3]);
        state.mDepthCubemapShaderProgram->setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc4,
                                                  state.mCubeViewProjMatrices[4]);
        state.mDepthCubemapShaderProgram->setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc5,
                                                  state.mCubeViewProjMatrices[5]);

        int modelIndex = 0;
        for (size_t i = 0; i < renderObjects.size(); i++)
        {
            if (renderObjects[i].instanced)
            {
                modelIndex += renderObjects[i].instanceCount;              
            }
            else
            {
                state.mDepthCubemapShaderProgram->setMat4(state.mDepthCubemapShaderModelLoc, models[modelIndex]);
                
                Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);
                modelIndex++;
            }
        }

        light->getNativeGraphicsShadowCubemapFBO()->unbind();
    }
}

static void renderOpaques(World *world, Camera *camera, Light *light, Transform *lightTransform,
                                  ForwardRendererState &state, const std::vector<RenderObject> &renderObjects, 
                                  const std::vector<glm::mat4> &models)
{
    state.mLightState.mPosition = lightTransform->getPosition();
    state.mLightState.mDirection = lightTransform->getForward();
    state.mLightState.mColor = light->mColor;

    state.mLightState.mIntensity = light->mIntensity;
    state.mLightState.mSpotAngle = light->mSpotAngle;
    state.mLightState.mInnerSpotAngle = light->mInnerSpotAngle;
    state.mLightState.mShadowStrength = light->mShadowStrength;
    state.mLightState.mShadowBias = light->mShadowBias;

    if (light->mLightType == LightType::Directional)
    {
        for (int i = 0; i < 5; i++)
        {
            state.mLightState.mLightProjection[i] = state.mCascadeOrthoProj[i];

            glm::vec4 cascadeEnd = glm::vec4(0.0f, 0.0f, state.mCascadeEnds[i + 1], 1.0f);
            glm::vec4 clipSpaceCascadeEnd = camera->getProjMatrix() * cascadeEnd;
            state.mLightState.mCascadeEnds[i] = clipSpaceCascadeEnd.z;

            state.mLightState.mLightView[i] = state.mCascadeLightView[i];
        }
    }
    else if (light->mLightType == LightType::Spot)
    {
        state.mLightState.mLightProjection[0] = state.mShadowProjMatrix;
        state.mLightState.mLightView[0] = state.mShadowViewMatrix;
    }

    Renderer::getRenderer()->setGlobalLightUniforms(state.mLightState);

    int64_t variant = static_cast<int64_t>(ShaderMacro::None);
    if (light->mLightType == LightType::Directional)
    {
        variant = static_cast<int64_t>(ShaderMacro::Directional);
        if (camera->mShadowCascades != ShadowCascades::NoCascades)
        {
            if (camera->mColorTarget == ColorTarget::ShadowCascades)
            {
                variant |= static_cast<int64_t>(ShaderMacro::ShowCascades);
            }
        }
    }
    else if (light->mLightType == LightType::Spot)
    {
        variant = static_cast<int64_t>(ShaderMacro::Spot);
    }
    else if (light->mLightType == LightType::Point)
    {
        variant = static_cast<int64_t>(ShaderMacro::Point);
    }

    if (light->mShadowType == ShadowType::Hard)
    {
        variant |= static_cast<int64_t>(ShaderMacro::HardShadows);
    }
    else if (light->mShadowType == ShadowType::Soft)
    {
        variant |= static_cast<int64_t>(ShaderMacro::SoftShadows);
    }

    Framebuffer *framebuffer = nullptr;

    if (camera->mRenderTextureId.isValid())
    {
        RenderTexture *renderTexture = world->getAssetByGuid<RenderTexture>(camera->mRenderTextureId);
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

    int currentMaterialIndex = -1;
    int currentShaderIndex = -1;

    Shader *shader = nullptr;
    Material *material = nullptr;

    int modelIndex = 0;
    for (size_t i = 0; i < renderObjects.size(); i++)
    {
        if (currentShaderIndex != renderObjects[i].shaderIndex)
        {
            shader = world->getAssetByIndex<Shader>(renderObjects[i].shaderIndex);
          
            if (renderObjects[i].instanced)
            {
                shader->bind(variant | static_cast<int64_t>(ShaderMacro::Instancing));
            }
            else
            {
                shader->bind(shader->getProgramFromVariant(variant) != nullptr ? variant : 0);
            }  

            currentShaderIndex = renderObjects[i].shaderIndex;
        }

        if (renderObjects[i].instanced)
        {
            Renderer::getRenderer()->updateInstanceBuffer(renderObjects[i].instanceModelVbo, &models[modelIndex],
                                           renderObjects[i].instanceCount);
            modelIndex += renderObjects[i].instanceCount;
        }
        else
        {
            shader->setMat4("model", models[modelIndex]);
            modelIndex++;
        }

        if (currentMaterialIndex != renderObjects[i].materialIndex)
        {
            material = world->getAssetByIndex<Material>(renderObjects[i].materialIndex);
            material->apply();

            currentMaterialIndex = renderObjects[i].materialIndex;
        }

        if (light->mLightType == LightType::Directional)
        {
            std::vector<TextureHandle*> tex = {light->getNativeGraphicsShadowCascadeFBO(0)->getDepthTex(),
                                               light->getNativeGraphicsShadowCascadeFBO(1)->getDepthTex(),
                                                light->getNativeGraphicsShadowCascadeFBO(2)->getDepthTex(),
                                                light->getNativeGraphicsShadowCascadeFBO(3)->getDepthTex(),
                                                light->getNativeGraphicsShadowCascadeFBO(4)->getDepthTex()};
            std::vector<int> texUnit = {3, 4, 5, 6, 7};
            shader->setTexture2Ds("shadowMap", texUnit, 5, tex);
        }
        else if (light->mLightType == LightType::Spot)
        {
            shader->setTexture2D("shadowMap[0]", 3, light->getNativeGrpahicsShadowSpotlightDepthTex());
        }

        if (renderObjects[i].instanced)
        {
            Renderer::getRenderer()->renderInstanced(renderObjects[i], camera->mQuery);
        }
        else
        {
            Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);
        }
    }

    framebuffer->unbind();
}

static void renderSprites(World *world, Camera *camera, ForwardRendererState &state,
                                  const std::vector<SpriteObject> &spriteObjects)
{
    state.mSpriteShaderProgram->bind();

    //float width = static_cast<float>(camera->getViewport().mWidth);
    //float height = static_cast<float>(camera->getViewport().mHeight);

    // glm::mat4 projection = glm::ortho(0.0f, width, 0.0f, height, -1.0f, 1.0f);
    glm::mat4 projection = camera->getProjMatrix();
    glm::mat4 view = camera->getViewMatrix();

    state.mSpriteShaderProgram->setMat4(state.mSpriteProjectionLoc, projection);
    state.mSpriteShaderProgram->setMat4(state.mSpriteViewLoc, view);

    Framebuffer *framebuffer = nullptr;

    if (camera->mRenderTextureId.isValid())
    {
        RenderTexture *renderTexture = world->getAssetByGuid<RenderTexture>(camera->mRenderTextureId);
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

    for (size_t i = 0; i < spriteObjects.size(); i++)
    {
        state.mSpriteShaderProgram->setMat4(state.mSpriteModelLoc, spriteObjects[i].model);
        state.mSpriteShaderProgram->setColor(state.mSpriteColorLoc, spriteObjects[i].color);
        state.mSpriteShaderProgram->setTexture2D(state.mSpriteImageLoc, 0, spriteObjects[i].texture);

        Renderer::getRenderer()->render(0, 6, spriteObjects[i].vao, camera->mQuery);
    }

    framebuffer->unbind();
}

static void renderColorPicking(World *world, Camera *camera, ForwardRendererState &state,
                                       const std::vector<RenderObject> &renderObjects,
                                       const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds)
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
    camera->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
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

            state.mColorShaderProgram->bind();
            state.mColorShaderProgram->setMat4(state.mColorShaderModelLoc, models[modelIndex]);
            state.mColorShaderProgram->setColor32(state.mColorShaderColorLoc, Color32(r, g, b, a));
            Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);

            modelIndex++;
        }
    }
   
    camera->getNativeGraphicsColorPickingFBO()->unbind();
}

static void renderTransparents()
{
}

static void postProcessing()
{
}

static void endFrame(World *world, Camera *camera, ForwardRendererState &state)
{
    if (camera->mRenderToScreen)
    {
        Renderer::getRenderer()->bindFramebuffer(0);
        Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                              camera->getViewport().mHeight);

        state.mQuadShaderProgram->bind();
        state.mQuadShaderProgram->setTexture2D(state.mQuadShaderTexLoc, 0, camera->getNativeGraphicsColorTex());

        Renderer::getRenderer()->renderScreenQuad(state.mQuadVAO);

        Renderer::getRenderer()->unbindFramebuffer();
    }

    camera->endQuery();

    Renderer::getRenderer()->turnOff(Capability::BackfaceCulling);
}

static void calcCascadeOrthoProj(Camera *camera, glm::vec3 lightDirection, ForwardRendererState &state)
{
    glm::mat4 viewInv = camera->getInvViewMatrix();
    float fov = camera->getFrustum().mFov;
    float aspect = camera->getFrustum().mAspectRatio;
    float tanHalfHFOV = glm::tan(glm::radians(0.5f * fov));
    float tanHalfVFOV = aspect * glm::tan(glm::radians(0.5f * fov));

    for (int i = 0; i < 5; i++)
    {
        float xn = -1.0f * state.mCascadeEnds[i] * tanHalfHFOV;
        float xf = -1.0f * state.mCascadeEnds[i + 1] * tanHalfHFOV;
        float yn = -1.0f * state.mCascadeEnds[i] * tanHalfVFOV;
        float yf = -1.0f * state.mCascadeEnds[i + 1] * tanHalfVFOV;

        // Find cascade frustum corners
        glm::vec4 frustumCorners[8];
        frustumCorners[0] = glm::vec4(xn, yn, state.mCascadeEnds[i], 1.0f);
        frustumCorners[1] = glm::vec4(-xn, yn, state.mCascadeEnds[i], 1.0f);
        frustumCorners[2] = glm::vec4(xn, -yn, state.mCascadeEnds[i], 1.0f);
        frustumCorners[3] = glm::vec4(-xn, -yn, state.mCascadeEnds[i], 1.0f);

        frustumCorners[4] = glm::vec4(xf, yf, state.mCascadeEnds[i + 1], 1.0f);
        frustumCorners[5] = glm::vec4(-xf, yf, state.mCascadeEnds[i + 1], 1.0f);
        frustumCorners[6] = glm::vec4(xf, -yf, state.mCascadeEnds[i + 1], 1.0f);
        frustumCorners[7] = glm::vec4(-xf, -yf, state.mCascadeEnds[i + 1], 1.0f);

        // find frustum centre by averaging corners
        glm::vec4 frustumCentre = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);

        frustumCentre += frustumCorners[0];
        frustumCentre += frustumCorners[1];
        frustumCentre += frustumCorners[2];
        frustumCentre += frustumCorners[3];
        frustumCentre += frustumCorners[4];
        frustumCentre += frustumCorners[5];
        frustumCentre += frustumCorners[6];
        frustumCentre += frustumCorners[7];

        frustumCentre *= 0.125f;
        frustumCentre.w = 1.0f;

        // Transform the frustum centre from view to world space
        glm::vec4 frustrumCentreWorldSpace = viewInv * frustumCentre;

        // need to set p such that it is far enough out to have the light projection capture all objects that
        // might cast shadows
        float d = std::max(80.0f, -(state.mCascadeEnds[i + 1] - state.mCascadeEnds[i]));
        glm::vec3 p = glm::vec3(frustrumCentreWorldSpace.x - d * lightDirection.x,
                                frustrumCentreWorldSpace.y - d * lightDirection.y,
                                frustrumCentreWorldSpace.z - d * lightDirection.z);

        state.mCascadeLightView[i] = glm::lookAt(p, glm::vec3(frustrumCentreWorldSpace), glm::vec3(0.0f, 1.0f, 0.0f));

        float minX = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::lowest();
        float minY = std::numeric_limits<float>::max();
        float maxY = std::numeric_limits<float>::lowest();
        float minZ = std::numeric_limits<float>::max();
        float maxZ = std::numeric_limits<float>::lowest();

        glm::mat4 cascadeLightView = state.mCascadeLightView[i];

        // Transform the frustum coordinates from view to world space and then world to light space
        glm::vec4 vL[8];
        for (int j = 0; j < 8; j++)
        {
            vL[j] = cascadeLightView * (viewInv * frustumCorners[j]);

            minX = glm::min(minX, vL[j].x);
            maxX = glm::max(maxX, vL[j].x);
            minY = glm::min(minY, vL[j].y);
            maxY = glm::max(maxY, vL[j].y);
            minZ = glm::min(minZ, vL[j].z);
            maxZ = glm::max(maxZ, vL[j].z);
        }
        // Should be glm::ortho(minX, maxX, minY, maxY, 0.0f, -minZ) but need to decrease/increase minZ and maxZ
        // respectively to capture all objects that can cast a show
        state.mCascadeOrthoProj[i] = glm::ortho(minX, maxX, minY, maxY, 0.0f, -minZ);
    }
}