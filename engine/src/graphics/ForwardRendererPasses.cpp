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

void PhysicsEngine::initializeRenderer(World *world, ForwardRendererState *state)
{
    // compile internal shader programs
    state->mGeometryShader = world->getAssetById<Shader>(world->getPositionAndNormalsShaderId());
    state->mColorShader = world->getAssetById<Shader>(world->getColorShaderId());
    state->mSsaoShader = world->getAssetById<Shader>(world->getSsaoShaderId());
    state->mDepthShader = world->getAssetById<Shader>(world->getShadowDepthMapShaderId());
    state->mDepthCubemapShader = world->getAssetById<Shader>(world->getShadowDepthCubemapShaderId());
    state->mQuadShader = world->getAssetById<Shader>(world->getScreenQuadShaderId());

    assert(state->mGeometryShader != NULL);
    assert(state->mColorShader != NULL);
    assert(state->mSsaoShader != NULL);
    assert(state->mDepthShader != NULL);
    assert(state->mDepthCubemapShader != NULL);
    assert(state->mQuadShader != NULL);

    state->mGeometryShader->compile();
    state->mColorShader->compile();
    state->mSsaoShader->compile();
    state->mDepthShader->compile();
    state->mDepthCubemapShader->compile();
    state->mQuadShader->compile();

    // cache internal shader uniforms
    state->mGeometryShaderProgram = state->mGeometryShader->getProgramFromVariant(ShaderVariant::None);
    state->mGeometryShaderModelLoc =
        state->mGeometryShader->findUniformLocation("model", state->mGeometryShaderProgram);

    state->mSsaoShaderProgram = state->mSsaoShader->getProgramFromVariant(ShaderVariant::None);
    state->mSsaoShaderProjectionLoc = state->mSsaoShader->findUniformLocation("projection", state->mSsaoShaderProgram);
    state->mSsaoShaderPositionTexLoc =
        state->mSsaoShader->findUniformLocation("positionTex", state->mSsaoShaderProgram);
    state->mSsaoShaderNormalTexLoc = state->mSsaoShader->findUniformLocation("normalTex", state->mSsaoShaderProgram);
    state->mSsaoShaderNoiseTexLoc = state->mSsaoShader->findUniformLocation("noiseTex", state->mSsaoShaderProgram);

    for (int i = 0; i < 64; i++)
    {
        state->mSsaoShaderSamplesLoc[i] =
            state->mSsaoShader->findUniformLocation("samples[" + std::to_string(i) + "]", state->mSsaoShaderProgram);
    }

    state->mDepthShaderProgram = state->mDepthShader->getProgramFromVariant(ShaderVariant::None);
    state->mDepthShaderModelLoc = state->mDepthShader->findUniformLocation("model", state->mDepthShaderProgram);
    state->mDepthShaderViewLoc = state->mDepthShader->findUniformLocation("view", state->mDepthShaderProgram);
    state->mDepthShaderProjectionLoc =
        state->mDepthShader->findUniformLocation("projection", state->mDepthShaderProgram);

    state->mDepthCubemapShaderProgram = state->mDepthCubemapShader->getProgramFromVariant(ShaderVariant::None);
    state->mDepthCubemapShaderLightPosLoc =
        state->mDepthCubemapShader->findUniformLocation("lightPos", state->mDepthCubemapShaderProgram);
    state->mDepthCubemapShaderFarPlaneLoc =
        state->mDepthCubemapShader->findUniformLocation("farPlane", state->mDepthCubemapShaderProgram);
    state->mDepthCubemapShaderModelLoc =
        state->mDepthCubemapShader->findUniformLocation("model", state->mDepthCubemapShaderProgram);
    state->mDepthCubemapShaderCubeViewProjMatricesLoc0 =
        state->mDepthCubemapShader->findUniformLocation("cubeViewProjMatrices[0]", state->mDepthCubemapShaderProgram);
    state->mDepthCubemapShaderCubeViewProjMatricesLoc1 =
        state->mDepthCubemapShader->findUniformLocation("cubeViewProjMatrices[1]", state->mDepthCubemapShaderProgram);
    state->mDepthCubemapShaderCubeViewProjMatricesLoc2 =
        state->mDepthCubemapShader->findUniformLocation("cubeViewProjMatrices[2]", state->mDepthCubemapShaderProgram);
    state->mDepthCubemapShaderCubeViewProjMatricesLoc3 =
        state->mDepthCubemapShader->findUniformLocation("cubeViewProjMatrices[3]", state->mDepthCubemapShaderProgram);
    state->mDepthCubemapShaderCubeViewProjMatricesLoc4 =
        state->mDepthCubemapShader->findUniformLocation("cubeViewProjMatrices[4]", state->mDepthCubemapShaderProgram);
    state->mDepthCubemapShaderCubeViewProjMatricesLoc5 =
        state->mDepthCubemapShader->findUniformLocation("cubeViewProjMatrices[5]", state->mDepthCubemapShaderProgram);

    state->mColorShaderProgram = state->mColorShader->getProgramFromVariant(ShaderVariant::None);
    state->mColorShaderModelLoc = state->mColorShader->findUniformLocation("model", state->mColorShaderProgram);
    state->mColorShaderColorLoc = state->mColorShader->findUniformLocation("color", state->mColorShaderProgram);

    Graphics::checkError(__LINE__, __FILE__);

    // generate screen quad for final rendering
    constexpr float quadVertices[] = {
        // positions        // texture Coords
        -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        1.0f,  1.0f, 0.0f, 1.0f, 1.0f, 1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,
    };

    glGenVertexArrays(1, &state->mQuadVAO);
    glBindVertexArray(state->mQuadVAO);

    glGenBuffers(1, &state->mQuadVBO);
    glBindBuffer(GL_ARRAY_BUFFER, state->mQuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    Graphics::checkError(__LINE__, __FILE__);

    glGenBuffers(1, &(state->mCameraState.mHandle));
    glBindBuffer(GL_UNIFORM_BUFFER, state->mCameraState.mHandle);
    glBufferData(GL_UNIFORM_BUFFER, 144, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glGenBuffers(1, &(state->mLightState.mHandle));
    glBindBuffer(GL_UNIFORM_BUFFER, state->mLightState.mHandle);
    glBufferData(GL_UNIFORM_BUFFER, 824, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::beginFrame(World *world, Camera *camera, ForwardRendererState *state)
{
    camera->beginQuery();

    state->mCameraState.mProjection = camera->getProjMatrix();
    state->mCameraState.mView = camera->getViewMatrix();
    state->mCameraState.mCameraPos = camera->getComponent<Transform>(world)->mPosition;

    // set camera state binding point and update camera state data
    glBindBuffer(GL_UNIFORM_BUFFER, state->mCameraState.mHandle);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, state->mCameraState.mHandle, 0, 144);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, 64, glm::value_ptr(state->mCameraState.mProjection));
    glBufferSubData(GL_UNIFORM_BUFFER, 64, 64, glm::value_ptr(state->mCameraState.mView));
    glBufferSubData(GL_UNIFORM_BUFFER, 128, 12, glm::value_ptr(state->mCameraState.mCameraPos));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    // set light state binding point
    glBindBuffer(GL_UNIFORM_BUFFER, state->mLightState.mHandle);
    glBindBufferRange(GL_UNIFORM_BUFFER, 1, state->mLightState.mHandle, 0, 824);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    // glEnable(GL_SCISSOR_TEST);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glDepthFunc(GL_LEQUAL);
    glBlendFunc(GL_ONE, GL_ZERO);
    glBlendEquation(GL_FUNC_ADD);

    glViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
               camera->getViewport().mHeight);
    glScissor(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
              camera->getViewport().mHeight);

    glClearColor(camera->mBackgroundColor.r, camera->mBackgroundColor.g, camera->mBackgroundColor.b,
                 camera->mBackgroundColor.a);
    glClearDepth(1.0f);

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsColorPickingFBO());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsColorPickingFBO());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsGeometryFBO());
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsSSAOFBO());
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PhysicsEngine::computeSSAO(World *world, Camera *camera, ForwardRendererState *state,
                                const std::vector<std::pair<uint64_t, int>> &renderQueue,
                                const std::vector<RenderObject> &renderObjects)
{
    // fill geometry framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsGeometryFBO());
    glViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
               camera->getViewport().mHeight);

    state->mGeometryShader->use(state->mGeometryShaderProgram);

    for (size_t i = 0; i < renderQueue.size(); i++)
    {
        state->mGeometryShader->setMat4(state->mGeometryShaderModelLoc, renderObjects[renderQueue[i].second].model);

        Graphics::render(renderObjects[renderQueue[i].second], camera->mQuery);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);

    // fill ssao color texture
    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsSSAOFBO());
    glViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
               camera->getViewport().mHeight);
    state->mSsaoShader->use(state->mSsaoShaderProgram);
    state->mSsaoShader->setMat4(state->mSsaoShaderProjectionLoc, camera->getProjMatrix());
    for (int i = 0; i < 64; i++)
    {
        state->mSsaoShader->setVec3(state->mSsaoShaderSamplesLoc[i], camera->getSSAOSample(i));
    }
    state->mSsaoShader->setInt(state->mSsaoShaderPositionTexLoc, 0);
    state->mSsaoShader->setInt(state->mSsaoShaderNormalTexLoc, 1);
    state->mSsaoShader->setInt(state->mSsaoShaderNoiseTexLoc, 2);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsPositionTex());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsNormalTex());
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsSSAONoiseTex());

    glBindVertexArray(state->mQuadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderShadows(World *world, Camera *camera, Light *light, Transform *lightTransform,
                                  ForwardRendererState *state, const std::vector<std::pair<uint64_t, int>> &renderQueue,
                                  const std::vector<RenderObject> &renderObjects)
{
    if (light->mLightType == LightType::Directional)
    {

        calcShadowmapCascades(camera, state);
        calcCascadeOrthoProj(camera, lightTransform->getForward(), state);

        for (int i = 0; i < 5; i++)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, light->getNativeGraphicsShadowCascadeFBO(i));
            glViewport(0, 0, static_cast<GLsizei>(light->getShadowMapResolution()),
                       static_cast<GLsizei>(light->getShadowMapResolution()));

            glClearDepth(1.0f);
            glClear(GL_DEPTH_BUFFER_BIT);

            state->mDepthShader->use(state->mDepthShaderProgram);
            state->mDepthShader->setMat4(state->mDepthShaderViewLoc, state->mCascadeLightView[i]);
            state->mDepthShader->setMat4(state->mDepthShaderProjectionLoc, state->mCascadeOrthoProj[i]);

            for (size_t j = 0; j < renderQueue.size(); j++)
            {
                state->mDepthShader->setMat4(state->mDepthShaderModelLoc, renderObjects[renderQueue[j].second].model);
                Graphics::render(renderObjects[renderQueue[j].second], camera->mQuery);
            }

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
    }
    else if (light->mLightType == LightType::Spot)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, light->getNativeGraphicsShadowSpotlightFBO());
        glViewport(0, 0, static_cast<GLsizei>(light->getShadowMapResolution()),
                   static_cast<GLsizei>(light->getShadowMapResolution()));

        glClearDepth(1.0f);
        glClear(GL_DEPTH_BUFFER_BIT);

        state->mShadowProjMatrix = light->getProjMatrix();
        state->mShadowViewMatrix =
            glm::lookAt(lightTransform->mPosition, lightTransform->mPosition + lightTransform->getForward(),
                        glm::vec3(0.0f, 1.0f, 0.0f));

        state->mDepthShader->use(state->mDepthShaderProgram);
        state->mDepthShader->setMat4(state->mDepthShaderProjectionLoc, state->mShadowProjMatrix);
        state->mDepthShader->setMat4(state->mDepthShaderViewLoc, state->mShadowViewMatrix);

        for (size_t i = 0; i < renderQueue.size(); i++)
        {
            state->mDepthShader->setMat4(state->mDepthShaderModelLoc, renderObjects[renderQueue[i].second].model);
            Graphics::render(renderObjects[renderQueue[i].second], camera->mQuery);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    else if (light->mLightType == LightType::Point)
    {

        state->mCubeViewProjMatrices[0] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition,
                                                  lightTransform->mPosition + glm::vec3(1.0, 0.0, 0.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));
        state->mCubeViewProjMatrices[1] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition,
                                                  lightTransform->mPosition + glm::vec3(-1.0, 0.0, 0.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));
        state->mCubeViewProjMatrices[2] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition,
                                                  lightTransform->mPosition + glm::vec3(0.0, 1.0, 0.0),
                                                  glm::vec3(0.0, 0.0, 1.0)));
        state->mCubeViewProjMatrices[3] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition,
                                                  lightTransform->mPosition + glm::vec3(0.0, -1.0, 0.0),
                                                  glm::vec3(0.0, 0.0, -1.0)));
        state->mCubeViewProjMatrices[4] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition,
                                                  lightTransform->mPosition + glm::vec3(0.0, 0.0, 1.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));
        state->mCubeViewProjMatrices[5] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->mPosition,
                                                  lightTransform->mPosition + glm::vec3(0.0, 0.0, -1.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));

        glBindFramebuffer(GL_FRAMEBUFFER, light->getNativeGraphicsShadowCubemapFBO());
        glViewport(0, 0, static_cast<GLsizei>(light->getShadowMapResolution()),
                   static_cast<GLsizei>(light->getShadowMapResolution()));

        glClearDepth(1.0f);
        glClear(GL_DEPTH_BUFFER_BIT);

        state->mDepthCubemapShader->use(state->mDepthCubemapShaderProgram);
        state->mDepthCubemapShader->setVec3(state->mDepthCubemapShaderLightPosLoc, lightTransform->mPosition);
        state->mDepthCubemapShader->setFloat(state->mDepthCubemapShaderFarPlaneLoc, camera->mFrustum.mFarPlane);
        state->mDepthCubemapShader->setMat4(state->mDepthCubemapShaderCubeViewProjMatricesLoc0,
                                            state->mCubeViewProjMatrices[0]);
        state->mDepthCubemapShader->setMat4(state->mDepthCubemapShaderCubeViewProjMatricesLoc1,
                                            state->mCubeViewProjMatrices[1]);
        state->mDepthCubemapShader->setMat4(state->mDepthCubemapShaderCubeViewProjMatricesLoc2,
                                            state->mCubeViewProjMatrices[2]);
        state->mDepthCubemapShader->setMat4(state->mDepthCubemapShaderCubeViewProjMatricesLoc3,
                                            state->mCubeViewProjMatrices[3]);
        state->mDepthCubemapShader->setMat4(state->mDepthCubemapShaderCubeViewProjMatricesLoc4,
                                            state->mCubeViewProjMatrices[4]);
        state->mDepthCubemapShader->setMat4(state->mDepthCubemapShaderCubeViewProjMatricesLoc5,
                                            state->mCubeViewProjMatrices[5]);

        for (size_t i = 0; i < renderQueue.size(); i++)
        {
            state->mDepthCubemapShader->setMat4(state->mDepthCubemapShaderModelLoc,
                                                renderObjects[renderQueue[i].second].model);
            Graphics::render(renderObjects[renderQueue[i].second], camera->mQuery);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

void PhysicsEngine::renderOpaques(World *world, Camera *camera, Light *light, Transform *lightTransform,
                                  ForwardRendererState *state, const std::vector<std::pair<uint64_t, int>> &renderQueue,
                                  const std::vector<RenderObject> &renderObjects)
{
    state->mLightState.mPosition = lightTransform->mPosition;
    state->mLightState.mDirection = lightTransform->getForward();
    state->mLightState.mColor = light->mColor;

    state->mLightState.mIntensity = light->mIntensity;
    state->mLightState.mSpotAngle = light->mSpotAngle;
    state->mLightState.mInnerSpotAngle = light->mInnerSpotAngle;
    state->mLightState.mShadowNearPlane = light->mShadowNearPlane;
    state->mLightState.mShadowFarPlane = light->mShadowFarPlane;
    state->mLightState.mShadowAngle = light->mShadowAngle;
    state->mLightState.mShadowRadius = light->mShadowRadius;
    state->mLightState.mShadowStrength = light->mShadowStrength;

    if (light->mLightType == LightType::Directional)
    {
        for (int i = 0; i < 5; i++)
        {
            state->mLightState.mLightProjection[i] = state->mCascadeOrthoProj[i];

            glm::vec4 cascadeEnd = glm::vec4(0.0f, 0.0f, state->mCascadeEnds[i + 1], 1.0f);
            glm::vec4 clipSpaceCascadeEnd = camera->getProjMatrix() * cascadeEnd;
            state->mLightState.mCascadeEnds[i] = clipSpaceCascadeEnd.z;

            state->mLightState.mLightView[i] = state->mCascadeLightView[i];
        }
    }
    else if (light->mLightType == LightType::Spot)
    {
        for (int i = 0; i < 5; i++)
        {
            state->mLightState.mLightProjection[i] = state->mShadowProjMatrix;
            state->mLightState.mLightView[i] = state->mShadowViewMatrix;
        }
    }

    glBindBuffer(GL_UNIFORM_BUFFER, state->mLightState.mHandle);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, 320, &state->mLightState.mLightProjection[0]);
    glBufferSubData(GL_UNIFORM_BUFFER, 320, 320, &state->mLightState.mLightView[0]);
    glBufferSubData(GL_UNIFORM_BUFFER, 640, 12, glm::value_ptr(state->mLightState.mPosition));
    glBufferSubData(GL_UNIFORM_BUFFER, 656, 12, glm::value_ptr(state->mLightState.mDirection));
    glBufferSubData(GL_UNIFORM_BUFFER, 672, 16, glm::value_ptr(state->mLightState.mColor));
    glBufferSubData(GL_UNIFORM_BUFFER, 688, 4, &state->mLightState.mCascadeEnds[0]);
    glBufferSubData(GL_UNIFORM_BUFFER, 704, 4, &state->mLightState.mCascadeEnds[1]);
    glBufferSubData(GL_UNIFORM_BUFFER, 720, 4, &state->mLightState.mCascadeEnds[2]);
    glBufferSubData(GL_UNIFORM_BUFFER, 736, 4, &state->mLightState.mCascadeEnds[3]);
    glBufferSubData(GL_UNIFORM_BUFFER, 752, 4, &state->mLightState.mCascadeEnds[4]);
    glBufferSubData(GL_UNIFORM_BUFFER, 768, 4, &state->mLightState.mIntensity);
    glBufferSubData(GL_UNIFORM_BUFFER, 772, 4, &(state->mLightState.mSpotAngle));
    glBufferSubData(GL_UNIFORM_BUFFER, 776, 4, &(state->mLightState.mInnerSpotAngle));
    glBufferSubData(GL_UNIFORM_BUFFER, 780, 4, &(state->mLightState.mShadowNearPlane));
    glBufferSubData(GL_UNIFORM_BUFFER, 784, 4, &(state->mLightState.mShadowFarPlane));
    glBufferSubData(GL_UNIFORM_BUFFER, 788, 4, &(state->mLightState.mShadowAngle));
    glBufferSubData(GL_UNIFORM_BUFFER, 792, 4, &(state->mLightState.mShadowRadius));
    glBufferSubData(GL_UNIFORM_BUFFER, 796, 4, &(state->mLightState.mShadowStrength));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

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

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());
    glViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
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
                shader->setInt(shaderShadowMapNames[j], 3 + j);

                glActiveTexture(GL_TEXTURE0 + 3 + j);
                glBindTexture(GL_TEXTURE_2D, light->getNativeGraphicsShadowCascadeDepthTex(j));
            }
        }
        else if (light->mLightType == LightType::Spot)
        {
            shader->setInt(shaderShadowMapNames[0], 3);

            glActiveTexture(GL_TEXTURE0 + 3);
            glBindTexture(GL_TEXTURE_2D, light->getNativeGrpahicsShadowSpotlightDepthTex());
        }

        Graphics::render(renderObjects[renderQueue[i].second], camera->mQuery);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderColorPicking(World *world, Camera *camera, ForwardRendererState *state,
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

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsColorPickingFBO());

    glViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
               camera->getViewport().mHeight);

    state->mColorShader->use(state->mColorShaderProgram);

    color = 1;
    for (size_t i = 0; i < renderQueue.size(); i++)
    {
        int r = (color & 0x000000FF) >> 0;
        int g = (color & 0x0000FF00) >> 8;
        int b = (color & 0x00FF0000) >> 16;

        color++;

        state->mColorShader->setMat4(state->mColorShaderModelLoc, renderObjects[renderQueue[i].second].model);
        state->mColorShader->setVec4(state->mColorShaderColorLoc, glm::vec4(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f));

        Graphics::render(renderObjects[renderQueue[i].second], camera->mQuery);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderTransparents()
{
}

void PhysicsEngine::postProcessing()
{
}

void PhysicsEngine::endFrame(World *world, Camera *camera, ForwardRendererState *state)
{
    camera->endQuery();

    /*if (state->mRenderToScreen) {
        glViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
    camera->getViewport().mHeight); glScissor(camera->getViewport().mX, camera->getViewport().mY,
    camera->getViewport().mWidth, camera->getViewport().mHeight);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        state->mQuadShader->use(ShaderVariant::None);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsColorTex());

        glBindVertexArray(state->mQuadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
    }*/
}

void PhysicsEngine::calcShadowmapCascades(Camera *camera, ForwardRendererState *state)
{
    float nearDist = camera->mFrustum.mNearPlane;
    float farDist = camera->mFrustum.mFarPlane;

    const float splitWeight = 0.95f;
    const float ratio = farDist / nearDist;

    for (int i = 0; i < 6; i++)
    {
        const float si = i / 5.0f;

        state->mCascadeEnds[i] = -1.0f * (splitWeight * (nearDist * powf(ratio, si)) +
                                          (1 - splitWeight) * (nearDist + (farDist - nearDist) * si));
    }
}

void PhysicsEngine::calcCascadeOrthoProj(Camera *camera, glm::vec3 lightDirection, ForwardRendererState *state)
{
    glm::mat4 viewInv = glm::inverse(camera->getViewMatrix());
    float fov = camera->mFrustum.mFov;
    float aspect = camera->mFrustum.mAspectRatio;
    float tanHalfHFOV = glm::tan(glm::radians(0.5f * fov));
    float tanHalfVFOV = glm::tan(glm::radians(0.5f * fov * aspect));

    for (int i = 0; i < 5; i++)
    {
        float xn = -1.0f * state->mCascadeEnds[i] * tanHalfHFOV;
        float xf = -1.0f * state->mCascadeEnds[i + 1] * tanHalfHFOV;
        float yn = -1.0f * state->mCascadeEnds[i] * tanHalfVFOV;
        float yf = -1.0f * state->mCascadeEnds[i + 1] * tanHalfVFOV;

        // Find cascade frustum corners
        glm::vec4 frustumCorners[8];
        frustumCorners[0] = glm::vec4(xn, yn, state->mCascadeEnds[i], 1.0f);
        frustumCorners[1] = glm::vec4(-xn, yn, state->mCascadeEnds[i], 1.0f);
        frustumCorners[2] = glm::vec4(xn, -yn, state->mCascadeEnds[i], 1.0f);
        frustumCorners[3] = glm::vec4(-xn, -yn, state->mCascadeEnds[i], 1.0f);

        frustumCorners[4] = glm::vec4(xf, yf, state->mCascadeEnds[i + 1], 1.0f);
        frustumCorners[5] = glm::vec4(-xf, yf, state->mCascadeEnds[i + 1], 1.0f);
        frustumCorners[6] = glm::vec4(xf, -yf, state->mCascadeEnds[i + 1], 1.0f);
        frustumCorners[7] = glm::vec4(-xf, -yf, state->mCascadeEnds[i + 1], 1.0f);

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

        state->mCascadeLightView[i] = glm::lookAt(p, glm::vec3(frustrumCentreWorldSpace), glm::vec3(1.0f, 0.0f, 0.0f));

        float minX = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::lowest();
        float minY = std::numeric_limits<float>::max();
        float maxY = std::numeric_limits<float>::lowest();
        float minZ = std::numeric_limits<float>::max();
        float maxZ = std::numeric_limits<float>::lowest();

        glm::mat4 cascadeLightView = state->mCascadeLightView[i];

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

        state->mCascadeOrthoProj[i] = glm::ortho(minX, maxX, minY, maxY, 0.0f, -minZ);
    }
}