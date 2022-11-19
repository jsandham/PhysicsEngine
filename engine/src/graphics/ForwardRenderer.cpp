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
    state.mDepthShaderProgram = RendererShaders::getRendererShaders()->getDepthShader().mProgram;
    state.mDepthShaderModelLoc = RendererShaders::getRendererShaders()->getDepthShader().mModelLoc;
    state.mDepthShaderViewLoc = RendererShaders::getRendererShaders()->getDepthShader().mViewLoc;
    state.mDepthShaderProjectionLoc = RendererShaders::getRendererShaders()->getDepthShader().mProjectionLoc;

    state.mDepthCubemapShaderProgram = RendererShaders::getRendererShaders()->getDepthCubemapShader().mProgram;
    state.mDepthCubemapShaderLightPosLoc = RendererShaders::getRendererShaders()->getDepthCubemapShader().mLightPosLoc;
    state.mDepthCubemapShaderFarPlaneLoc = RendererShaders::getRendererShaders()->getDepthCubemapShader().mFarPlaneLoc;
    state.mDepthCubemapShaderModelLoc = RendererShaders::getRendererShaders()->getDepthCubemapShader().mModelLoc;
    state.mDepthCubemapShaderCubeViewProjMatricesLoc0 = RendererShaders::getRendererShaders()->getDepthCubemapShader().mCubeViewProjMatricesLoc0;
    state.mDepthCubemapShaderCubeViewProjMatricesLoc1 = RendererShaders::getRendererShaders()->getDepthCubemapShader().mCubeViewProjMatricesLoc1;
    state.mDepthCubemapShaderCubeViewProjMatricesLoc2 = RendererShaders::getRendererShaders()->getDepthCubemapShader().mCubeViewProjMatricesLoc2;
    state.mDepthCubemapShaderCubeViewProjMatricesLoc3 = RendererShaders::getRendererShaders()->getDepthCubemapShader().mCubeViewProjMatricesLoc3;
    state.mDepthCubemapShaderCubeViewProjMatricesLoc4 = RendererShaders::getRendererShaders()->getDepthCubemapShader().mCubeViewProjMatricesLoc4;
    state.mDepthCubemapShaderCubeViewProjMatricesLoc5 = RendererShaders::getRendererShaders()->getDepthCubemapShader().mCubeViewProjMatricesLoc5;

    state.mGeometryShaderProgram = RendererShaders::getRendererShaders()->getGeometryShader().mProgram;
    state.mGeometryShaderModelLoc = RendererShaders::getRendererShaders()->getGeometryShader().mModelLoc;

    state.mColorShaderProgram = RendererShaders::getRendererShaders()->getColorShader().mProgram;
    state.mColorShaderModelLoc = RendererShaders::getRendererShaders()->getColorShader().mModelLoc;
    state.mColorShaderColorLoc = RendererShaders::getRendererShaders()->getColorShader().mColorLoc;
    state.mColorInstancedShaderProgram = RendererShaders::getRendererShaders()->getColorInstancedShader().mProgram;

    state.mSsaoShaderProgram = RendererShaders::getRendererShaders()->getSSAOShader().mProgram;
    state.mSsaoShaderProjectionLoc = RendererShaders::getRendererShaders()->getSSAOShader().mProjectionLoc;
    state.mSsaoShaderPositionTexLoc = RendererShaders::getRendererShaders()->getSSAOShader().mPositionTexLoc;
    state.mSsaoShaderNormalTexLoc = RendererShaders::getRendererShaders()->getSSAOShader().mNormalTexLoc;
    state.mSsaoShaderNoiseTexLoc = RendererShaders::getRendererShaders()->getSSAOShader().mNoiseTexLoc;
    for (int i = 0; i < 64; i++)
    {
        state.mSsaoShaderSamplesLoc[i] = RendererShaders::getRendererShaders()->getSSAOShader().mSamplesLoc[i];
    }

    state.mSpriteShaderProgram = RendererShaders::getRendererShaders()->getSpriteShader().mProgram;
    state.mSpriteModelLoc = RendererShaders::getRendererShaders()->getSpriteShader().mModelLoc;
    state.mSpriteViewLoc = RendererShaders::getRendererShaders()->getSpriteShader().mViewLoc;
    state.mSpriteProjectionLoc = RendererShaders::getRendererShaders()->getSpriteShader().mProjectionLoc;
    state.mSpriteColorLoc = RendererShaders::getRendererShaders()->getSpriteShader().mColorLoc;
    state.mSpriteImageLoc = RendererShaders::getRendererShaders()->getSpriteShader().mImageLoc;

    state.mQuadShaderProgram = RendererShaders::getRendererShaders()->getScreenQuadShader().mProgram;
    state.mQuadShaderTexLoc = RendererShaders::getRendererShaders()->getScreenQuadShader().mTexLoc;

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

    if (camera->mRenderTextureId.isValid())
    {
        RenderTexture *renderTexture = world->getAssetByGuid<RenderTexture>(camera->mRenderTextureId);
        if (renderTexture != nullptr)
        {
            Renderer::getRenderer()->bindFramebuffer(renderTexture->getNativeGraphicsMainFBO());
        }
        else
        {
            Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
        }
    }
    else
    {
        Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
    }
    Renderer::getRenderer()->clearFrambufferColor(camera->mBackgroundColor);
    Renderer::getRenderer()->clearFramebufferDepth(1.0f);
    Renderer::getRenderer()->unbindFramebuffer();

    Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsColorPickingFBO());
    Renderer::getRenderer()->clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
    Renderer::getRenderer()->clearFramebufferDepth(1.0f);
    Renderer::getRenderer()->unbindFramebuffer();

    if(camera->mSSAO == CameraSSAO::SSAO_On)
    {
        Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsSSAOFBO());
        Renderer::getRenderer()->clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
        Renderer::getRenderer()->unbindFramebuffer();
    }
}

static void computeSSAO(World *world, Camera *camera, ForwardRendererState &state,
                                const std::vector<RenderObject> &renderObjects, const std::vector<glm::mat4> &models)
{
    // fill geometry framebuffer
    Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsGeometryFBO());
    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Renderer::getRenderer()->use(state.mGeometryShaderProgram);

    int modelIndex = 0;
    for (size_t i = 0; i < renderObjects.size(); i++)
    {
        if (renderObjects[i].instanced)
        {
            modelIndex += renderObjects[i].instanceCount;
        }
        else
        {
            Renderer::getRenderer()->setMat4(state.mGeometryShaderModelLoc, models[modelIndex]);
            Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);
            modelIndex++;
        }
    }

    Renderer::getRenderer()->unbindFramebuffer();

    // fill ssao color texture
    Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsSSAOFBO());
    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Renderer::getRenderer()->use(state.mSsaoShaderProgram);
    Renderer::getRenderer()->setMat4(state.mSsaoShaderProjectionLoc, camera->getProjMatrix());
    for (int i = 0; i < 64; i++)
    {
        Renderer::getRenderer()->setVec3(state.mSsaoShaderSamplesLoc[i], camera->getSSAOSample(i));
    }
    Renderer::getRenderer()->setTexture2D(state.mSsaoShaderPositionTexLoc, 0, camera->getNativeGraphicsPositionTex());
    Renderer::getRenderer()->setTexture2D(state.mSsaoShaderNormalTexLoc, 1, camera->getNativeGraphicsNormalTex());
    Renderer::getRenderer()->setTexture2D(state.mSsaoShaderNoiseTexLoc, 2, camera->getNativeGraphicsSSAONoiseTex());

    Renderer::getRenderer()->renderScreenQuad(state.mQuadVAO);

    Renderer::getRenderer()->unbindFramebuffer();
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
            Renderer::getRenderer()->bindFramebuffer(light->getNativeGraphicsShadowCascadeFBO(i));
            Renderer::getRenderer()->setViewport(0, 0, static_cast<int>(light->getShadowMapResolution()),
                                  static_cast<int>(light->getShadowMapResolution()));

            Renderer::getRenderer()->clearFramebufferDepth(1.0f);

            Renderer::getRenderer()->use(state.mDepthShaderProgram);
            Renderer::getRenderer()->setMat4(state.mDepthShaderViewLoc, state.mCascadeLightView[i]);
            Renderer::getRenderer()->setMat4(state.mDepthShaderProjectionLoc, state.mCascadeOrthoProj[i]);

            int modelIndex = 0;
            for (size_t j = 0; j < renderObjects.size(); j++)
            {
                if (renderObjects[j].instanced)
                {
                    modelIndex += renderObjects[j].instanceCount;                    
                }
                else
                {
                    Renderer::getRenderer()->setMat4(state.mDepthShaderModelLoc, models[modelIndex]);
                    Renderer::getRenderer()->render(renderObjects[j], camera->mQuery);
                    modelIndex++;
                }
            }

            //RendererShaders::getRendererShaders()->getDepthShader()->getModelLocation();
            //RendererShaders::getRendererShaders()->getDepthShader()->getViewLocation();
            //RendererShaders::getRendererShaders()->getDepthShader()->getProjectionLocation();
            //RendererShaders::getRendererShaders()->getDepthShader()->getVertexShader();
            //RendererShaders::getRendererShaders()->getDepthShader()->getFragmentShader();
            //shader mDepthShader;
            //mDepthShader.get

            Renderer::getRenderer()->unbindFramebuffer();
        }
    }
    else if (light->mLightType == LightType::Spot)
    {
        Renderer::getRenderer()->bindFramebuffer(light->getNativeGraphicsShadowSpotlightFBO());
        Renderer::getRenderer()->setViewport(0, 0, static_cast<int>(light->getShadowMapResolution()),
                              static_cast<int>(light->getShadowMapResolution()));

        Renderer::getRenderer()->clearFramebufferDepth(1.0f);

        state.mShadowProjMatrix = light->getProjMatrix();
        state.mShadowViewMatrix =
            glm::lookAt(lightTransform->getPosition(), lightTransform->getPosition() + lightTransform->getForward(),
                        glm::vec3(0.0f, 1.0f, 0.0f));

        Renderer::getRenderer()->use(state.mDepthShaderProgram);
        Renderer::getRenderer()->setMat4(state.mDepthShaderProjectionLoc, state.mShadowProjMatrix);
        Renderer::getRenderer()->setMat4(state.mDepthShaderViewLoc, state.mShadowViewMatrix);

        int modelIndex = 0;
        for (size_t i = 0; i < renderObjects.size(); i++)
        {
            if (renderObjects[i].instanced)
            {
                modelIndex += renderObjects[i].instanceCount;
            }
            else
            {
                Renderer::getRenderer()->setMat4(state.mDepthShaderModelLoc, models[modelIndex]);
                Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);
                modelIndex++;
            }
        }

        Renderer::getRenderer()->unbindFramebuffer();
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

        Renderer::getRenderer()->bindFramebuffer(light->getNativeGraphicsShadowCubemapFBO());
        Renderer::getRenderer()->setViewport(0, 0, static_cast<int>(light->getShadowMapResolution()),
                              static_cast<int>(light->getShadowMapResolution()));

        Renderer::getRenderer()->clearFramebufferDepth(1.0f);

        Renderer::getRenderer()->use(state.mDepthCubemapShaderProgram);
        Renderer::getRenderer()->setVec3(state.mDepthCubemapShaderLightPosLoc, lightTransform->getPosition());
        Renderer::getRenderer()->setFloat(state.mDepthCubemapShaderFarPlaneLoc,
                           camera->getFrustum().mFarPlane); // shadow map far plane?
        Renderer::getRenderer()->setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc0, state.mCubeViewProjMatrices[0]);
        Renderer::getRenderer()->setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc1, state.mCubeViewProjMatrices[1]);
        Renderer::getRenderer()->setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc2, state.mCubeViewProjMatrices[2]);
        Renderer::getRenderer()->setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc3, state.mCubeViewProjMatrices[3]);
        Renderer::getRenderer()->setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc4, state.mCubeViewProjMatrices[4]);
        Renderer::getRenderer()->setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc5, state.mCubeViewProjMatrices[5]);

        int modelIndex = 0;
        for (size_t i = 0; i < renderObjects.size(); i++)
        {
            if (renderObjects[i].instanced)
            {
                modelIndex += renderObjects[i].instanceCount;              
            }
            else
            {
                Renderer::getRenderer()->setMat4(state.mDepthCubemapShaderModelLoc, models[modelIndex]);
                Renderer::getRenderer()->render(renderObjects[i], camera->mQuery);
                modelIndex++;
            }
        }

        Renderer::getRenderer()->unbindFramebuffer();
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

    if (camera->mRenderTextureId.isValid())
    {
        RenderTexture *renderTexture = world->getAssetByGuid<RenderTexture>(camera->mRenderTextureId);
        if (renderTexture != nullptr)
        {
            Renderer::getRenderer()->bindFramebuffer(renderTexture->getNativeGraphicsMainFBO());
        }
        else
        {
            Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
        }
    }
    else
    {
        Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
    }

    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
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
                shader->use(shader->getProgramFromVariant(variant | static_cast<int64_t>(ShaderMacro::Instancing))); 
            }
            else
            {
                shader->use(shader->getProgramFromVariant(variant) != -1 ? shader->getProgramFromVariant(variant)
                                                                         : shader->getProgramFromVariant(0));
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
            int tex[5] = {light->getNativeGraphicsShadowCascadeDepthTex(0),
                                   light->getNativeGraphicsShadowCascadeDepthTex(1),
                                   light->getNativeGraphicsShadowCascadeDepthTex(2), 
                                   light->getNativeGraphicsShadowCascadeDepthTex(3),
                                   light->getNativeGraphicsShadowCascadeDepthTex(4)};
            int texUnit[5] = {3, 4, 5, 6, 7};
            shader->setTexture2Ds("shadowMap", &texUnit[0], 5, &tex[0]);
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

    Renderer::getRenderer()->unbindFramebuffer();
}

static void renderSprites(World *world, Camera *camera, ForwardRendererState &state,
                                  const std::vector<SpriteObject> &spriteObjects)
{
    Renderer::getRenderer()->use(state.mSpriteShaderProgram);

    //float width = static_cast<float>(camera->getViewport().mWidth);
    //float height = static_cast<float>(camera->getViewport().mHeight);

    // glm::mat4 projection = glm::ortho(0.0f, width, 0.0f, height, -1.0f, 1.0f);
    glm::mat4 projection = camera->getProjMatrix();
    glm::mat4 view = camera->getViewMatrix();

    Renderer::getRenderer()->setMat4(state.mSpriteProjectionLoc, projection);
    Renderer::getRenderer()->setMat4(state.mSpriteViewLoc, view);

    if (camera->mRenderTextureId.isValid())
    {
        RenderTexture *renderTexture = world->getAssetByGuid<RenderTexture>(camera->mRenderTextureId);
        if (renderTexture != nullptr)
        {
            Renderer::getRenderer()->bindFramebuffer(renderTexture->getNativeGraphicsMainFBO());
        }
        else
        {
            Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
        }
    }
    else
    {
        Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
    }

    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    for (size_t i = 0; i < spriteObjects.size(); i++)
    {
        Renderer::getRenderer()->setMat4(state.mSpriteModelLoc, spriteObjects[i].model);
        Renderer::getRenderer()->setColor(state.mSpriteColorLoc, spriteObjects[i].color);
        Renderer::getRenderer()->setTexture2D(state.mSpriteImageLoc, 0, spriteObjects[i].texture);

        Renderer::getRenderer()->render(0, 6, spriteObjects[i].vao, camera->mQuery);
    }

    Renderer::getRenderer()->unbindFramebuffer();
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

    Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsColorPickingFBO());
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
   
    Renderer::getRenderer()->unbindFramebuffer();
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

        Renderer::getRenderer()->use(state.mQuadShaderProgram);
        Renderer::getRenderer()->setTexture2D(state.mQuadShaderTexLoc, 0, camera->getNativeGraphicsColorTex());

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