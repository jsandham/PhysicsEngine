#include "../../include/graphics/DeferredRendererPasses.h"
#include "../../include/graphics/Graphics.h"

#include "../../include/components/Light.h"

#include "../../include/core/InternalShaders.h"
#include "../../include/core/Shader.h"

using namespace PhysicsEngine;

void PhysicsEngine::initializeDeferredRenderer(World *world, DeferredRendererState &state)
{
    // generate all internal shader programs
    state.mGeometryShader.setVertexShader(InternalShaders::gbufferVertexShader);
    state.mGeometryShader.setFragmentShader(InternalShaders::gbufferFragmentShader);
    state.mGeometryShader.compile();

    state.mSimpleLitDeferredShader.setVertexShader(InternalShaders::simpleLitDeferredVertexShader);
    state.mSimpleLitDeferredShader.setFragmentShader(InternalShaders::simpleLitDeferredFragmentShader);
    state.mSimpleLitDeferredShader.compile();

    // cache internal shader uniforms
    state.mGeometryShaderProgram = state.mGeometryShader.getProgramFromVariant(ShaderVariant::None);
    state.mGeometryShaderModelLoc = state.mGeometryShader.findUniformLocation("model", state.mGeometryShaderProgram);
    state.mGeometryShaderDiffuseTexLoc =
        state.mGeometryShader.findUniformLocation("texture_diffuse1", state.mGeometryShaderProgram);
    state.mGeometryShaderSpecTexLoc =
        state.mGeometryShader.findUniformLocation("texture_specular1", state.mGeometryShaderProgram);

    state.mSimpleLitDeferredShaderProgram = state.mSimpleLitDeferredShader.getProgramFromVariant(ShaderVariant::None);
    state.mSimpleLitDeferredShaderViewPosLoc =
        state.mSimpleLitDeferredShader.findUniformLocation("viewPos", state.mSimpleLitDeferredShaderProgram);
    for (int i = 0; i < 32; i++)
    {
        state.mSimpleLitDeferredShaderLightLocs[i] = state.mSimpleLitDeferredShader.findUniformLocation(
            "lights [" + std::to_string(i) + "]", state.mSimpleLitDeferredShaderProgram);
    }

    // generate screen quad for final rendering
    constexpr float quadVertices[] = {
        // positions        // texture Coords
        -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        1.0f,  1.0f, 0.0f, 1.0f, 1.0f, 1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,
    };

    // screen quad mesh
    glGenVertexArrays(1, &state.mQuadVAO);
    glBindVertexArray(state.mQuadVAO);

    glGenBuffers(1, &state.mQuadVBO);
    glBindBuffer(GL_ARRAY_BUFFER, state.mQuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    Graphics::checkError(__LINE__, __FILE__);

    glGenBuffers(1, &(state.mCameraState.mBuffer));
    glBindBuffer(GL_UNIFORM_BUFFER, state.mCameraState.mBuffer);
    glBufferData(GL_UNIFORM_BUFFER, 144, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::beginDeferredFrame(World *world, Camera *camera, DeferredRendererState &state)
{
    camera->beginQuery();

    state.mCameraState.mProjection = camera->getProjMatrix();
    state.mCameraState.mView = camera->getViewMatrix();
    state.mCameraState.mCameraPos = camera->getComponent<Transform>(world)->mPosition;

    // set camera state binding point and update camera state data
    glBindBuffer(GL_UNIFORM_BUFFER, state.mCameraState.mBuffer);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, state.mCameraState.mBuffer, 0, 144);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, 64, glm::value_ptr(state.mCameraState.mProjection));
    glBufferSubData(GL_UNIFORM_BUFFER, 64, 64, glm::value_ptr(state.mCameraState.mView));
    glBufferSubData(GL_UNIFORM_BUFFER, 128, 12, glm::value_ptr(state.mCameraState.mCameraPos));
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glClearColor(camera->mBackgroundColor.r, camera->mBackgroundColor.g, camera->mBackgroundColor.b,
                 camera->mBackgroundColor.a);
    glClearDepth(1.0f);

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsGeometryFBO());
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PhysicsEngine::geometryPass(World *world, Camera *camera, DeferredRendererState &state,
                                 const std::vector<RenderObject> &renderObjects)
{
    // fill geometry framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsGeometryFBO());
    state.mGeometryShader.use(state.mGeometryShaderProgram);
    for (size_t i = 0; i < renderObjects.size(); i++)
    {
        state.mGeometryShader.setMat4(state.mGeometryShaderModelLoc, renderObjects[i].model);

        Graphics::render(renderObjects[i], camera->mQuery);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::lightingPass(World *world, Camera *camera, DeferredRendererState &state,
                                 const std::vector<RenderObject> &renderObjects)
{
    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());

    state.mSimpleLitDeferredShader.use(state.mSimpleLitDeferredShaderProgram);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsPositionTex());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsNormalTex());
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsAlbedoSpecTex());

    for (size_t i = 0; i < world->getNumberOfComponents<Light>(); i++)
    {
        // Light *light = world->getComponentByIndex<Light>(i);
        // Transform *lightTransform = light->getComponent<Transform>(world);

        // state->mSimpleLitDeferredShader.setVec3(state->mSimpleLitDeferredShaderLightPosLocs,
        // lightTransform->mPosition);
        // state->mSimpleLitDeferredShader.setVec3(state->mSimpleLitDeferredShaderLightColLocs, light->mAmbient);
    }

    glBindVertexArray(state.mQuadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PhysicsEngine::endDeferredFrame(World *world, Camera *camera, DeferredRendererState &state)
{
    camera->endQuery();

    if (state.mRenderToScreen)
    {
        glViewport(0, 0, 1024, 1024);
        glScissor(0, 0, 1024, 1024);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        state.mQuadShader.use(ShaderVariant::None);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsColorTex());

        glBindVertexArray(state.mQuadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
    }
}