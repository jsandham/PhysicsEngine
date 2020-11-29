#include <random>
#include <unordered_set>

#include "../../include/core/InternalShaders.h"
#include "../../include/core/Shader.h"

#include "../../include/graphics/ForwardRendererPasses.h"
#include "../../include/graphics/Graphics.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"
#include "../glm/gtc/type_ptr.hpp"

using namespace PhysicsEngine;

void PhysicsEngine::initializeRenderer(World *world, ForwardRendererState &state)
{
    // compile internal shader programs
    Shader *mGeometryShader = world->getAssetById<Shader>(world->getPositionAndNormalsShaderId());
    Shader *mColorShader = world->getAssetById<Shader>(world->getColorShaderId());
    Shader *mSsaoShader = world->getAssetById<Shader>(world->getSsaoShaderId());
    Shader *mDepthShader = world->getAssetById<Shader>(world->getShadowDepthMapShaderId());
    Shader *mDepthCubemapShader = world->getAssetById<Shader>(world->getShadowDepthCubemapShaderId());
    Shader *mQuadShader = world->getAssetById<Shader>(world->getScreenQuadShaderId());

    assert(mGeometryShader != NULL);
    assert(mColorShader != NULL);
    assert(mSsaoShader != NULL);
    assert(mDepthShader != NULL);
    assert(mDepthCubemapShader != NULL);
    assert(mQuadShader != NULL);

    mGeometryShader->compile();
    mColorShader->compile();
    mSsaoShader->compile();
    mDepthShader->compile();
    mDepthCubemapShader->compile();
    mQuadShader->compile();

    // cache internal shader uniforms
    state.mGeometryShaderProgram = mGeometryShader->getProgramFromVariant(ShaderVariant::None);
    state.mGeometryShaderModelLoc = mGeometryShader->findUniformLocation("model", state.mGeometryShaderProgram);

    state.mSsaoShaderProgram = mSsaoShader->getProgramFromVariant(ShaderVariant::None);
    state.mSsaoShaderProjectionLoc = mSsaoShader->findUniformLocation("projection", state.mSsaoShaderProgram);
    state.mSsaoShaderPositionTexLoc = mSsaoShader->findUniformLocation("positionTex", state.mSsaoShaderProgram);
    state.mSsaoShaderNormalTexLoc = mSsaoShader->findUniformLocation("normalTex", state.mSsaoShaderProgram);
    state.mSsaoShaderNoiseTexLoc = mSsaoShader->findUniformLocation("noiseTex", state.mSsaoShaderProgram);

    for (int i = 0; i < 64; i++)
    {
        state.mSsaoShaderSamplesLoc[i] =
            mSsaoShader->findUniformLocation("samples[" + std::to_string(i) + "]", state.mSsaoShaderProgram);
    }

    state.mDepthShaderProgram = mDepthShader->getProgramFromVariant(ShaderVariant::None);
    state.mDepthShaderModelLoc = mDepthShader->findUniformLocation("model", state.mDepthShaderProgram);
    state.mDepthShaderViewLoc = mDepthShader->findUniformLocation("view", state.mDepthShaderProgram);
    state.mDepthShaderProjectionLoc = mDepthShader->findUniformLocation("projection", state.mDepthShaderProgram);

    state.mDepthCubemapShaderProgram = mDepthCubemapShader->getProgramFromVariant(ShaderVariant::None);
    state.mDepthCubemapShaderLightPosLoc =
        mDepthCubemapShader->findUniformLocation("lightPos", state.mDepthCubemapShaderProgram);
    state.mDepthCubemapShaderFarPlaneLoc =
        mDepthCubemapShader->findUniformLocation("farPlane", state.mDepthCubemapShaderProgram);
    state.mDepthCubemapShaderModelLoc =
        mDepthCubemapShader->findUniformLocation("model", state.mDepthCubemapShaderProgram);
    state.mDepthCubemapShaderCubeViewProjMatricesLoc0 =
        mDepthCubemapShader->findUniformLocation("cubeViewProjMatrices[0]", state.mDepthCubemapShaderProgram);
    state.mDepthCubemapShaderCubeViewProjMatricesLoc1 =
        mDepthCubemapShader->findUniformLocation("cubeViewProjMatrices[1]", state.mDepthCubemapShaderProgram);
    state.mDepthCubemapShaderCubeViewProjMatricesLoc2 =
        mDepthCubemapShader->findUniformLocation("cubeViewProjMatrices[2]", state.mDepthCubemapShaderProgram);
    state.mDepthCubemapShaderCubeViewProjMatricesLoc3 =
        mDepthCubemapShader->findUniformLocation("cubeViewProjMatrices[3]", state.mDepthCubemapShaderProgram);
    state.mDepthCubemapShaderCubeViewProjMatricesLoc4 =
        mDepthCubemapShader->findUniformLocation("cubeViewProjMatrices[4]", state.mDepthCubemapShaderProgram);
    state.mDepthCubemapShaderCubeViewProjMatricesLoc5 =
        mDepthCubemapShader->findUniformLocation("cubeViewProjMatrices[5]", state.mDepthCubemapShaderProgram);

    state.mColorShaderProgram = mColorShader->getProgramFromVariant(ShaderVariant::None);
    state.mColorShaderModelLoc = mColorShader->findUniformLocation("model", state.mColorShaderProgram);
    state.mColorShaderColorLoc = mColorShader->findUniformLocation("material.color", state.mColorShaderProgram);

    Graphics::createScreenQuad(&state.mQuadVAO, &state.mQuadVBO);

    Graphics::createGlobalCameraUniforms(state.mCameraState);
    Graphics::createGlobalLightUniforms(state.mLightState);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glDepthFunc(GL_LEQUAL);
    glBlendFunc(GL_ONE, GL_ZERO);
    glBlendEquation(GL_FUNC_ADD);
}

void PhysicsEngine::beginFrame(World *world, Camera *camera, ForwardRendererState &state)
{
    camera->beginQuery();

    state.mCameraState.mProjection = camera->getProjMatrix();
    state.mCameraState.mView = camera->getViewMatrix();
    state.mCameraState.mCameraPos = camera->getComponent<Transform>(world)->mPosition;

    // update camera state data
    Graphics::setGlobalCameraUniforms(state.mCameraState);

    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Graphics::bindFramebuffer(camera->getNativeGraphicsMainFBO());
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
        calcShadowmapCascades(camera, state);
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
        Graphics::setFloat(state.mDepthCubemapShaderFarPlaneLoc, camera->getFrustum().mFarPlane);
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
    state.mLightState.mShadowNearPlane = light->mShadowNearPlane;
    state.mLightState.mShadowFarPlane = light->mShadowFarPlane;
    state.mLightState.mShadowAngle = light->mShadowAngle;
    state.mLightState.mShadowRadius = light->mShadowRadius;
    state.mLightState.mShadowStrength = light->mShadowStrength;

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
        for (int i = 0; i < 5; i++)
        {
            state.mLightState.mLightProjection[i] = state.mShadowProjMatrix;
            state.mLightState.mLightView[i] = state.mShadowViewMatrix;
        }
    }

    Graphics::setGlobalLightUniforms(state.mLightState);

    int variant = ShaderVariant::None;
    if (light->mLightType == LightType::Directional)
    {
        variant = ShaderVariant::Directional;
    }
    else if (light->mLightType == LightType::Spot)
    {
        variant = ShaderVariant::Spot;
    }
    else if (light->mLightType == LightType::Point)
    {
        variant = ShaderVariant::Point;
    }

    if (light->mShadowType == ShadowType::Hard)
    {
        variant |= ShaderVariant::HardShadows;
    }
    else if (light->mShadowType == ShadowType::Soft)
    {
        variant |= ShaderVariant::SoftShadows;
    }

    const char *const shaderShadowMapNames[] = {"shadowMap[0]", "shadowMap[1]", "shadowMap[2]", "shadowMap[3]",
                                                "shadowMap[4]"};

    Graphics::bindFramebuffer(camera->getNativeGraphicsMainFBO());
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

void PhysicsEngine::renderColorPicking(World *world, Camera *camera, ForwardRendererState &state,
                                       const std::vector<std::pair<uint64_t, int>> &renderQueue,
                                       const std::vector<RenderObject> &renderObjects)
{
    camera->clearColoring();

    // assign colors to render objects.
    int color = 1;
    for (size_t i = 0; i < renderQueue.size(); i++)
    {
        camera->assignColoring(color, renderObjects[renderQueue[i].second].transformId);

        color++;
    }

    Graphics::bindFramebuffer(camera->getNativeGraphicsColorPickingFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Graphics::use(state.mColorShaderProgram);

    color = 1;
    for (size_t i = 0; i < renderQueue.size(); i++)
    {
        int r = (color & 0x000000FF) >> 0;
        int g = (color & 0x0000FF00) >> 8;
        int b = (color & 0x00FF0000) >> 16;

        color++;

        Graphics::setMat4(state.mColorShaderModelLoc, renderObjects[renderQueue[i].second].model);
        Graphics::setVec4(state.mColorShaderColorLoc, glm::vec4(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f));

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
    camera->endQuery();
}

void PhysicsEngine::calcShadowmapCascades(Camera *camera, ForwardRendererState &state)
{
    float nearDist = camera->getFrustum().mNearPlane;
    float farDist = camera->getFrustum().mFarPlane;

    const float splitWeight = 0.95f;
    const float ratio = farDist / nearDist;

    for (int i = 0; i < 6; i++)
    {
        const float si = i / 5.0f;

        state.mCascadeEnds[i] = -1.0f * (splitWeight * (nearDist * powf(ratio, si)) +
                                         (1 - splitWeight) * (nearDist + (farDist - nearDist) * si));
    }
}

void PhysicsEngine::calcCascadeOrthoProj(Camera *camera, glm::vec3 lightDirection, ForwardRendererState &state)
{
    glm::mat4 viewInv = camera->getInvViewMatrix();
    float fov = camera->getFrustum().mFov;
    float aspect = camera->getFrustum().mAspectRatio;
    float tanHalfHFOV = glm::tan(glm::radians(0.5f * fov));
    float tanHalfVFOV = glm::tan(glm::radians(0.5f * fov * aspect));

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
        float d = 40.0f; // cascadeEnds[i + 1] - cascadeEnds[i];

        glm::vec3 p = glm::vec3(frustrumCentreWorldSpace.x + d * lightDirection.x,
                                frustrumCentreWorldSpace.y + d * lightDirection.y,
                                frustrumCentreWorldSpace.z + d * lightDirection.z);

        state.mCascadeLightView[i] = glm::lookAt(p, glm::vec3(frustrumCentreWorldSpace), glm::vec3(1.0f, 0.0f, 0.0f));

        float minX = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::lowest();
        float minY = std::numeric_limits<float>::max();
        float maxY = std::numeric_limits<float>::lowest();
        float minZ = std::numeric_limits<float>::max();
        float maxZ = std::numeric_limits<float>::lowest();

        glm::mat4 cascadeLightView = state.mCascadeLightView[i];

        // Transform the frustum coordinates from view to world space and then world to light space
        glm::vec4 vL0 = cascadeLightView * (viewInv * frustumCorners[0]);
        glm::vec4 vL1 = cascadeLightView * (viewInv * frustumCorners[1]);
        glm::vec4 vL2 = cascadeLightView * (viewInv * frustumCorners[2]);
        glm::vec4 vL3 = cascadeLightView * (viewInv * frustumCorners[3]);
        glm::vec4 vL4 = cascadeLightView * (viewInv * frustumCorners[4]);
        glm::vec4 vL5 = cascadeLightView * (viewInv * frustumCorners[5]);
        glm::vec4 vL6 = cascadeLightView * (viewInv * frustumCorners[6]);
        glm::vec4 vL7 = cascadeLightView * (viewInv * frustumCorners[7]);

        minX = glm::min(minX, vL0.x);
        maxX = glm::max(maxX, vL0.x);
        minY = glm::min(minY, vL0.y);
        maxY = glm::max(maxY, vL0.y);
        minZ = glm::min(minZ, vL0.z);
        maxZ = glm::max(maxZ, vL0.z);

        minX = glm::min(minX, vL1.x);
        maxX = glm::max(maxX, vL1.x);
        minY = glm::min(minY, vL1.y);
        maxY = glm::max(maxY, vL1.y);
        minZ = glm::min(minZ, vL1.z);
        maxZ = glm::max(maxZ, vL1.z);

        minX = glm::min(minX, vL2.x);
        maxX = glm::max(maxX, vL2.x);
        minY = glm::min(minY, vL2.y);
        maxY = glm::max(maxY, vL2.y);
        minZ = glm::min(minZ, vL2.z);
        maxZ = glm::max(maxZ, vL2.z);

        minX = glm::min(minX, vL3.x);
        maxX = glm::max(maxX, vL3.x);
        minY = glm::min(minY, vL3.y);
        maxY = glm::max(maxY, vL3.y);
        minZ = glm::min(minZ, vL3.z);
        maxZ = glm::max(maxZ, vL3.z);

        minX = glm::min(minX, vL4.x);
        maxX = glm::max(maxX, vL4.x);
        minY = glm::min(minY, vL4.y);
        maxY = glm::max(maxY, vL4.y);
        minZ = glm::min(minZ, vL4.z);
        maxZ = glm::max(maxZ, vL4.z);

        minX = glm::min(minX, vL5.x);
        maxX = glm::max(maxX, vL5.x);
        minY = glm::min(minY, vL5.y);
        maxY = glm::max(maxY, vL5.y);
        minZ = glm::min(minZ, vL5.z);
        maxZ = glm::max(maxZ, vL5.z);

        minX = glm::min(minX, vL6.x);
        maxX = glm::max(maxX, vL6.x);
        minY = glm::min(minY, vL6.y);
        maxY = glm::max(maxY, vL6.y);
        minZ = glm::min(minZ, vL6.z);
        maxZ = glm::max(maxZ, vL6.z);

        minX = glm::min(minX, vL7.x);
        maxX = glm::max(maxX, vL7.x);
        minY = glm::min(minY, vL7.y);
        maxY = glm::max(maxY, vL7.y);
        minZ = glm::min(minZ, vL7.z);
        maxZ = glm::max(maxZ, vL7.z);

        state.mCascadeOrthoProj[i] = glm::ortho(minX, maxX, minY, maxY, 0.0f, -minZ);
    }
}