#include "../../include/graphics/ForwardRenderer.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

ForwardRenderer::ForwardRenderer()
{
}

ForwardRenderer::~ForwardRenderer()
{
}

void ForwardRenderer::init(World *world, bool renderToScreen)
{
    mWorld = world;
    mState.mRenderToScreen = renderToScreen;

    initializeRenderer(mWorld, mState);
}

void ForwardRenderer::update(const Input &input, Camera *camera,
                             const std::vector<std::pair<uint64_t, int>> &renderQueue,
                             const std::vector<RenderObject> &renderObjects,
                             const std::vector<SpriteObject> &spriteObjects)
{
    beginFrame(mWorld, camera, mState);

    if (camera->mSSAO == CameraSSAO::SSAO_On)
    {
        computeSSAO(mWorld, camera, mState, renderQueue, renderObjects);
    }

    for (size_t j = 0; j < mWorld->getNumberOfComponents<Light>(); j++)
    {
        Light *light = mWorld->getComponentByIndex<Light>(j);
        
        if (light->mEnabled)
        {
            Transform* lightTransform = light->getComponent<Transform>();

            if (lightTransform != nullptr)
            {
                renderShadows(mWorld, camera, light, lightTransform, mState, renderQueue, renderObjects);
                renderOpaques(mWorld, camera, light, lightTransform, mState, renderQueue, renderObjects);
                renderTransparents();
            }
        }
    }

    renderSprites(mWorld, camera, mState, spriteObjects);

    renderColorPicking(mWorld, camera, mState, renderQueue, renderObjects);

    postProcessing();

    endFrame(mWorld, camera, mState);
}

void PhysicsEngine::initializeRenderer(World *world, ForwardRendererState &state)
{
    Graphics::compileSSAOShader(state);
    Graphics::compileShadowDepthMapShader(state);
    Graphics::compileShadowDepthCubemapShader(state);
    Graphics::compileColorShader(state);
    Graphics::compileScreenQuadShader(state);
    Graphics::compileSpriteShader(state);

    Graphics::createScreenQuad(&state.mQuadVAO, &state.mQuadVBO);

    Graphics::createGlobalCameraUniforms(state.mCameraState);
    Graphics::createGlobalLightUniforms(state.mLightState);

    Graphics::turnOn(Capability::Depth_Testing);
    Graphics::turnOn(Capability::Blending);
    glDepthFunc(GL_LEQUAL);
    glBlendFunc(GL_ONE, GL_ZERO);
    glBlendEquation(GL_FUNC_ADD);
}

void PhysicsEngine::beginFrame(World *world, Camera *camera, ForwardRendererState &state)
{
    camera->beginQuery();

    state.mCameraState.mProjection = camera->getProjMatrix();
    state.mCameraState.mView = camera->getViewMatrix();
    state.mCameraState.mCameraPos = camera->getComponent<Transform>()->mPosition;

    // update camera state data
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

    Graphics::bindFramebuffer(camera->getNativeGraphicsColorPickingFBO());
    Graphics::clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
    Graphics::clearFramebufferDepth(1.0f);
    Graphics::unbindFramebuffer();

    Graphics::bindFramebuffer(camera->getNativeGraphicsGeometryFBO());
    Graphics::clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
    Graphics::unbindFramebuffer();

    Graphics::bindFramebuffer(camera->getNativeGraphicsSSAOFBO());
    Graphics::clearFrambufferColor(0.0f, 0.0f, 0.0f, 1.0f);
    Graphics::unbindFramebuffer();
}

void PhysicsEngine::computeSSAO(World *world, Camera *camera, ForwardRendererState &state,
                                const std::vector<std::pair<uint64_t, int>> &renderQueue,
                                const std::vector<RenderObject> &renderObjects)
{
    // fill geometry framebuffer
    Graphics::bindFramebuffer(camera->getNativeGraphicsGeometryFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Graphics::use(state.mGeometryShaderProgram);

    for (size_t i = 0; i < renderQueue.size(); i++)
    {
        Graphics::setMat4(state.mGeometryShaderModelLoc, renderObjects[renderQueue[i].second].model);

        Graphics::render(renderObjects[renderQueue[i].second], camera->mQuery);
    }

    Graphics::unbindFramebuffer();

    Graphics::checkError(__LINE__, __FILE__);

    // fill ssao color texture
    Graphics::bindFramebuffer(camera->getNativeGraphicsSSAOFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Graphics::use(state.mSsaoShaderProgram);
    Graphics::setMat4(state.mSsaoShaderProjectionLoc, camera->getProjMatrix());
    for (int i = 0; i < 64; i++)
    {
        Graphics::setVec3(state.mSsaoShaderSamplesLoc[i], camera->getSSAOSample(i));
    }
    Graphics::setTexture2D(state.mSsaoShaderPositionTexLoc, 0, camera->getNativeGraphicsPositionTex());
    Graphics::setTexture2D(state.mSsaoShaderNormalTexLoc, 1, camera->getNativeGraphicsNormalTex());
    Graphics::setTexture2D(state.mSsaoShaderNoiseTexLoc, 2, camera->getNativeGraphicsSSAONoiseTex());

    Graphics::renderScreenQuad(state.mQuadVAO);

    Graphics::unbindFramebuffer();

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderShadows(World *world, Camera *camera, Light *light, Transform *lightTransform,
                                  ForwardRendererState &state, const std::vector<std::pair<uint64_t, int>> &renderQueue,
                                  const std::vector<RenderObject> &renderObjects)
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
            Graphics::bindFramebuffer(light->getNativeGraphicsShadowCascadeFBO(i));
            Graphics::setViewport(0, 0, static_cast<int>(light->getShadowMapResolution()),
                                  static_cast<int>(light->getShadowMapResolution()));

            Graphics::clearFramebufferDepth(1.0f);

            Graphics::use(state.mDepthShaderProgram);
            Graphics::setMat4(state.mDepthShaderViewLoc, state.mCascadeLightView[i]);
            Graphics::setMat4(state.mDepthShaderProjectionLoc, state.mCascadeOrthoProj[i]);

            for (size_t j = 0; j < renderQueue.size(); j++)
            {
                Graphics::setMat4(state.mDepthShaderModelLoc, renderObjects[renderQueue[j].second].model);
                Graphics::render(renderObjects[renderQueue[j].second], camera->mQuery);
            }

            Graphics::unbindFramebuffer();
        }
    }
    else if (light->mLightType == LightType::Spot)
    {
        Graphics::bindFramebuffer(light->getNativeGraphicsShadowSpotlightFBO());
        Graphics::setViewport(0, 0, static_cast<int>(light->getShadowMapResolution()),
                              static_cast<int>(light->getShadowMapResolution()));

        Graphics::clearFramebufferDepth(1.0f);

        state.mShadowProjMatrix = light->getProjMatrix();
        state.mShadowViewMatrix =
            glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + lightTransform->getForward(),
                        glm::vec3(0.0f, 1.0f, 0.0f));

        Graphics::use(state.mDepthShaderProgram);
        Graphics::setMat4(state.mDepthShaderProjectionLoc, state.mShadowProjMatrix);
        Graphics::setMat4(state.mDepthShaderViewLoc, state.mShadowViewMatrix);

        for (size_t i = 0; i < renderQueue.size(); i++)
        {
            Graphics::setMat4(state.mDepthShaderModelLoc, renderObjects[renderQueue[i].second].model);
            Graphics::render(renderObjects[renderQueue[i].second], camera->mQuery);
        }

        Graphics::unbindFramebuffer();
    }
    else if (light->mLightType == LightType::Point)
    {

        state.mCubeViewProjMatrices[0] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition,
                                                  lightTransform->mPosition + glm::vec3(1.0, 0.0, 0.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));
        state.mCubeViewProjMatrices[1] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition,
                                                  lightTransform->mPosition + glm::vec3(-1.0, 0.0, 0.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));
        state.mCubeViewProjMatrices[2] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition,
                                                  lightTransform->mPosition + glm::vec3(0.0, 1.0, 0.0),
                                                  glm::vec3(0.0, 0.0, 1.0)));
        state.mCubeViewProjMatrices[3] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition,
                                                  lightTransform->mPosition + glm::vec3(0.0, -1.0, 0.0),
                                                  glm::vec3(0.0, 0.0, -1.0)));
        state.mCubeViewProjMatrices[4] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition,
                                                  lightTransform->mPosition + glm::vec3(0.0, 0.0, 1.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));
        state.mCubeViewProjMatrices[5] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition,
                                                  lightTransform->mPosition + glm::vec3(0.0, 0.0, -1.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));

        Graphics::bindFramebuffer(light->getNativeGraphicsShadowCubemapFBO());
        Graphics::setViewport(0, 0, static_cast<int>(light->getShadowMapResolution()),
                              static_cast<int>(light->getShadowMapResolution()));

        Graphics::clearFramebufferDepth(1.0f);

        Graphics::use(state.mDepthCubemapShaderProgram);
        Graphics::setVec3(state.mDepthCubemapShaderLightPosLoc, lightTransform->mPosition);
        Graphics::setFloat(state.mDepthCubemapShaderFarPlaneLoc,
                           camera->getFrustum().mFarPlane); // shadow map far plane?
        Graphics::setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc0, state.mCubeViewProjMatrices[0]);
        Graphics::setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc1, state.mCubeViewProjMatrices[1]);
        Graphics::setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc2, state.mCubeViewProjMatrices[2]);
        Graphics::setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc3, state.mCubeViewProjMatrices[3]);
        Graphics::setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc4, state.mCubeViewProjMatrices[4]);
        Graphics::setMat4(state.mDepthCubemapShaderCubeViewProjMatricesLoc5, state.mCubeViewProjMatrices[5]);

        for (size_t i = 0; i < renderQueue.size(); i++)
        {
            Graphics::setMat4(state.mDepthCubemapShaderModelLoc, renderObjects[renderQueue[i].second].model);
            Graphics::render(renderObjects[renderQueue[i].second], camera->mQuery);
        }

        Graphics::unbindFramebuffer();
    }
}

void PhysicsEngine::renderOpaques(World *world, Camera *camera, Light *light, Transform *lightTransform,
                                  ForwardRendererState &state, const std::vector<std::pair<uint64_t, int>> &renderQueue,
                                  const std::vector<RenderObject> &renderObjects)
{
    state.mLightState.mPosition = lightTransform->mPosition;
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

    Graphics::setGlobalLightUniforms(state.mLightState);

    int64_t variant = static_cast<int64_t>(ShaderMacro::None);
    if (light->mLightType == LightType::Directional)
    {
        variant = static_cast<int64_t>(ShaderMacro::Directional);
        if (camera->mShadowCascades != ShadowCascades::NoCascades)
        {
            variant |= static_cast<int64_t>(ShaderMacro::ShowCascades);
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

    const char *const shaderShadowMapNames[] = {"shadowMap[0]", "shadowMap[1]", "shadowMap[2]", "shadowMap[3]",
                                                "shadowMap[4]"};

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

    int currentMaterialIndex = -1;
    int currentShaderIndex = -1;

    Shader *shader = NULL;
    Material *material = NULL;

    for (size_t i = 0; i < renderQueue.size(); i++)
    {
        if (currentShaderIndex != renderObjects[renderQueue[i].second].shaderIndex)
        {
            shader = world->getAssetByIndex<Shader>(renderObjects[renderQueue[i].second].shaderIndex);
            shader->use(shader->getProgramFromVariant(variant));

            currentShaderIndex = renderObjects[renderQueue[i].second].shaderIndex;
        }

        shader->setMat4("model", renderObjects[renderQueue[i].second].model);

        if (currentMaterialIndex != renderObjects[renderQueue[i].second].materialIndex)
        {
            material = world->getAssetByIndex<Material>(renderObjects[renderQueue[i].second].materialIndex);
            material->apply(world);

            currentMaterialIndex = renderObjects[renderQueue[i].second].materialIndex;
        }

        if (light->mLightType == LightType::Directional)
        {
            for (int j = 0; j < 5; j++)
            {
                shader->setTexture2D(shaderShadowMapNames[j], 3 + j, light->getNativeGraphicsShadowCascadeDepthTex(j));
            }
        }
        else if (light->mLightType == LightType::Spot)
        {
            shader->setTexture2D(shaderShadowMapNames[0], 3, light->getNativeGrpahicsShadowSpotlightDepthTex());
        }

        Graphics::render(renderObjects[renderQueue[i].second], camera->mQuery);
    }

    Graphics::unbindFramebuffer();

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderSprites(World *world, Camera *camera, ForwardRendererState &state,
                                  const std::vector<SpriteObject> &spriteObjects)
{
    Graphics::use(state.mSpriteShaderProgram);

    float width = static_cast<float>(camera->getViewport().mWidth);
    float height = static_cast<float>(camera->getViewport().mHeight);

    // glm::mat4 projection = glm::ortho(0.0f, width, 0.0f, height, -1.0f, 1.0f);
    glm::mat4 projection = camera->getProjMatrix();
    glm::mat4 view = camera->getViewMatrix();

    Graphics::setMat4(state.mSpriteProjectionLoc, projection);
    Graphics::setMat4(state.mSpriteViewLoc, view);

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

    for (size_t i = 0; i < spriteObjects.size(); i++)
    {
        Graphics::setMat4(state.mSpriteModelLoc, spriteObjects[i].model);
        Graphics::setColor(state.mSpriteColorLoc, spriteObjects[i].color);
        Graphics::setTexture2D(state.mSpriteImageLoc, 0, spriteObjects[i].texture);

        Graphics::render(0, 6, spriteObjects[i].vao);
    }

    Graphics::unbindFramebuffer();
}

void PhysicsEngine::renderColorPicking(World *world, Camera *camera, ForwardRendererState &state,
                                       const std::vector<std::pair<uint64_t, int>> &renderQueue,
                                       const std::vector<RenderObject> &renderObjects)
{
    camera->clearColoring();

    // assign colors to render objects.
    uint32_t color = 1;
    for (size_t i = 0; i < renderQueue.size(); i++)
    {
        unsigned char r = 255 - ((color & 0x000000FF) >> 0);
        unsigned char g = 255 - ((color & 0x0000FF00) >> 8);
        unsigned char b = 255 - ((color & 0x00FF0000) >> 16);
        unsigned char a = 255;

        camera->assignColoring(Color32(r, g, b, a), renderObjects[renderQueue[i].second].transformId);

        color++;
    }

    Graphics::bindFramebuffer(camera->getNativeGraphicsColorPickingFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Graphics::use(state.mColorShaderProgram);

    color = 1;
    for (size_t i = 0; i < renderQueue.size(); i++)
    {
        unsigned char r = 255 - ((color & 0x000000FF) >> 0);
        unsigned char g = 255 - ((color & 0x0000FF00) >> 8);
        unsigned char b = 255 - ((color & 0x00FF0000) >> 16);
        unsigned char a = 255;

        color++;

        Graphics::setMat4(state.mColorShaderModelLoc, renderObjects[renderQueue[i].second].model);
        Graphics::setColor32(state.mColorShaderColorLoc, Color32(r, g, b, a));

        Graphics::render(renderObjects[renderQueue[i].second], camera->mQuery);
    }

    Graphics::unbindFramebuffer();

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderTransparents()
{
}

void PhysicsEngine::postProcessing()
{
}

void PhysicsEngine::endFrame(World *world, Camera *camera, ForwardRendererState &state)
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

void PhysicsEngine::calcCascadeOrthoProj(Camera *camera, glm::vec3 lightDirection, ForwardRendererState &state)
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