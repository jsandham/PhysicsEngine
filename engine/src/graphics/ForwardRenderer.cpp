#include "../../include/graphics/ForwardRenderer.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

ForwardRenderer::ForwardRenderer()
{
}

ForwardRenderer::~ForwardRenderer()
{
}

void ForwardRenderer::init(World *world)
{
    mWorld = world;

    mQuadShader = RendererShaders::getScreenQuadShader();
    mDepthShader = RendererShaders::getDepthShader();
    mDepthCubemapShader = RendererShaders::getDepthCubemapShader();
    mGeometryShader = RendererShaders::getGeometryShader();
    mColorShader = RendererShaders::getColorShader();
    mColorInstancedShader = RendererShaders::getColorInstancedShader();
    mSsaoShader = RendererShaders::getSSAOShader();

    mCameraUniform = RendererUniforms::getCameraUniform();
    mLightUniform = RendererUniforms::getLightUniform();

    mScreenQuad = RendererMeshes::getScreenQuad();

    Renderer::getRenderer()->turnOn(Capability::Depth_Testing);
}

void ForwardRenderer::update(Camera *camera, const std::vector<DrawCallCommand> &commands,
                             const std::vector<glm::mat4> &models, const std::vector<Id> &transformIds, OcclusionQuery &occlusionQuery)
{
    beginFrame(camera);

    if (camera->mSSAO == CameraSSAO::SSAO_On)
    {
        computeSSAO(camera, commands, models);
    }

    for (size_t j = 0; j < mWorld->getActiveScene()->getNumberOfComponents<Light>(); j++)
    {
        Light *light = mWorld->getActiveScene()->getComponentByIndex<Light>(j);

        if (light->mEnabled)
        {
            Transform *lightTransform = light->getComponent<Transform>();

            if (lightTransform != nullptr)
            {
                if (light->mShadowType != ShadowType::None)
                {
                    renderShadows(camera, light, lightTransform, commands, models);
                }

                renderOpaques(camera, light, lightTransform, commands, models, occlusionQuery);

                renderTransparents();
            }
        }
    }

    renderColorPicking(camera, commands, models, transformIds);

    postProcessing();

    endFrame(camera);
}

void ForwardRenderer::beginFrame(Camera *camera)
{
    Renderer::getRenderer()->turnOn(Capability::BackfaceCulling);

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

    if (camera->mSSAO == CameraSSAO::SSAO_On)
    {
        camera->getNativeGraphicsSSAOFBO()->bind();
        camera->getNativeGraphicsSSAOFBO()->clearColor(Color::black);
        camera->getNativeGraphicsSSAOFBO()->unbind();
    }
}

void ForwardRenderer::computeSSAO(Camera *camera, const std::vector<DrawCallCommand> &commands,
                                  const std::vector<glm::mat4> &models)
{
    // fill geometry framebuffer
    camera->getNativeGraphicsGeometryFBO()->bind();
    camera->getNativeGraphicsGeometryFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                        camera->getViewport().mWidth, camera->getViewport().mHeight);

    mGeometryShader->bind();

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
            modelIndex += Renderer::INSTANCE_BATCH_SIZE;
        }
        else
        {
            mGeometryShader->setModel(models[modelIndex]);

            Renderer::getRenderer()->drawIndexed(mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
                                                 (subMeshVertexEndIndex - subMeshVertexStartIndex),
                                                 camera->mQuery);
            modelIndex++;
        }
    }

    camera->getNativeGraphicsGeometryFBO()->unbind();

    // fill ssao color texture
    camera->getNativeGraphicsSSAOFBO()->bind();
    camera->getNativeGraphicsSSAOFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                    camera->getViewport().mWidth, camera->getViewport().mHeight);

    mSsaoShader->bind();
    mSsaoShader->setProjection(camera->getProjMatrix());
    for (int i = 0; i < 64; i++)
    {
        mSsaoShader->setSample(i, camera->getSSAOSample(i));
    }
    mSsaoShader->setPositionTexture(0, camera->getNativeGraphicsPositionTex());
    mSsaoShader->setNormalTexture(1, camera->getNativeGraphicsNormalTex());
    mSsaoShader->setNoiseTexture(2, camera->getNativeGraphicsSSAONoiseTex());

    mScreenQuad->draw();

    camera->getNativeGraphicsSSAOFBO()->unbind();
}

void ForwardRenderer::renderShadows(Camera *camera, Light *light, Transform *lightTransform,
                                    const std::vector<DrawCallCommand> &commands,
                                    const std::vector<glm::mat4> &models)
{
    if (light->mLightType == LightType::Directional)
    {
        std::array<float, 6> cascadeEnds = camera->calcViewSpaceCascadeEnds();
        for (size_t i = 0; i < cascadeEnds.size(); i++)
        {
            mCascadeEnds[i] = cascadeEnds[i];
        }

        calcCascadeOrthoProj(camera, lightTransform->getForward());

        for (int i = 0; i < 5; i++)
        {
            light->getNativeGraphicsShadowCascadeFBO(i)->bind();
            light->getNativeGraphicsShadowCascadeFBO(i)->setViewport(0, 0,
                                                                     static_cast<int>(light->getShadowMapResolution()),
                                                                     static_cast<int>(light->getShadowMapResolution()));
            light->getNativeGraphicsShadowCascadeFBO(i)->clearDepth(1.0f);

            mDepthShader->bind();
            mDepthShader->setView(mCascadeLightView[i]);
            mDepthShader->setProjection(mCascadeOrthoProj[i]);

            int modelIndex = 0;
            for (size_t j = 0; j < commands.size(); j++)
            {
                uint16_t meshIndex = commands[j].getMeshIndex();
                uint8_t subMesh = commands[j].getSubMesh();
                bool instanced = commands[j].isInstanced();

                Mesh *mesh = mWorld->getAssetByIndex<Mesh>(meshIndex);

                int subMeshVertexStartIndex = mesh->getSubMeshStartIndex(subMesh);
                int subMeshVertexEndIndex = mesh->getSubMeshEndIndex(subMesh);

                if (instanced)
                {
                    modelIndex += Renderer::INSTANCE_BATCH_SIZE;
                }
                else
                {
                    mDepthShader->setModel(models[modelIndex]);
                    Renderer::getRenderer()->drawIndexed(mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
                                                         (subMeshVertexEndIndex - subMeshVertexStartIndex),
                                                         camera->mQuery);
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

        mShadowProjMatrix = light->getProjMatrix();
        mShadowViewMatrix =
            glm::lookAt(lightTransform->getPosition(), lightTransform->getPosition() + lightTransform->getForward(),
                        glm::vec3(0.0f, 1.0f, 0.0f));

        mDepthShader->bind();
        mDepthShader->setView(mShadowViewMatrix);
        mDepthShader->setProjection(mShadowProjMatrix);

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
                modelIndex += Renderer::INSTANCE_BATCH_SIZE;
            }
            else
            {
                mDepthShader->setModel(models[modelIndex]);

                Renderer::getRenderer()->drawIndexed(mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
                                                     (subMeshVertexEndIndex - subMeshVertexStartIndex),
                                                     camera->mQuery);
                modelIndex++;
            }
        }

        light->getNativeGraphicsShadowSpotlightFBO()->unbind();
    }
    else if (light->mLightType == LightType::Point)
    {

        mCubeViewProjMatrices[0] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->getPosition(),
                                                  lightTransform->getPosition() + glm::vec3(1.0, 0.0, 0.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));
        mCubeViewProjMatrices[1] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->getPosition(),
                                                  lightTransform->getPosition() + glm::vec3(-1.0, 0.0, 0.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));
        mCubeViewProjMatrices[2] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->getPosition(),
                                                  lightTransform->getPosition() + glm::vec3(0.0, 1.0, 0.0),
                                                  glm::vec3(0.0, 0.0, 1.0)));
        mCubeViewProjMatrices[3] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->getPosition(),
                                                  lightTransform->getPosition() + glm::vec3(0.0, -1.0, 0.0),
                                                  glm::vec3(0.0, 0.0, -1.0)));
        mCubeViewProjMatrices[4] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->getPosition(),
                                                  lightTransform->getPosition() + glm::vec3(0.0, 0.0, 1.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));
        mCubeViewProjMatrices[5] =
            (light->getProjMatrix() * glm::lookAt(lightTransform->getPosition(),
                                                  lightTransform->getPosition() + glm::vec3(0.0, 0.0, -1.0),
                                                  glm::vec3(0.0, -1.0, 0.0)));

        light->getNativeGraphicsShadowCubemapFBO()->bind();
        light->getNativeGraphicsShadowCubemapFBO()->setViewport(0, 0, static_cast<int>(light->getShadowMapResolution()),
                                                                static_cast<int>(light->getShadowMapResolution()));
        light->getNativeGraphicsShadowCubemapFBO()->clearDepth(1.0f);

        mDepthCubemapShader->bind();
        mDepthCubemapShader->setLightPos(lightTransform->getPosition());
        mDepthCubemapShader->setFarPlane(camera->getFrustum().mFarPlane);
        mDepthCubemapShader->setCubeViewProj(0, mCubeViewProjMatrices[0]);
        mDepthCubemapShader->setCubeViewProj(1, mCubeViewProjMatrices[1]);
        mDepthCubemapShader->setCubeViewProj(2, mCubeViewProjMatrices[2]);
        mDepthCubemapShader->setCubeViewProj(3, mCubeViewProjMatrices[3]);
        mDepthCubemapShader->setCubeViewProj(4, mCubeViewProjMatrices[4]);
        mDepthCubemapShader->setCubeViewProj(5, mCubeViewProjMatrices[5]);

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
                modelIndex += Renderer::INSTANCE_BATCH_SIZE;
            }
            else
            {
                mDepthCubemapShader->setModel(models[modelIndex]);

                Renderer::getRenderer()->drawIndexed(mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
                                                     (subMeshVertexEndIndex - subMeshVertexStartIndex),
                                                     camera->mQuery);
                modelIndex++;
            }
        }

        light->getNativeGraphicsShadowCubemapFBO()->unbind();
    }
}

void ForwardRenderer::renderOpaques(Camera *camera, Light *light, Transform *lightTransform,
                                    const std::vector<DrawCallCommand> &commands,
                                    const std::vector<glm::mat4> &models,
                                    OcclusionQuery &occlusionQuery)
{
    // Configure light uniform
    mLightUniform->setLightPosition(lightTransform->getPosition());
    mLightUniform->setLightDirection(lightTransform->getForward());
    mLightUniform->setLightColor(light->mColor);
    mLightUniform->setLightIntensity(light->mIntensity);
    mLightUniform->setSpotLightAngle(light->mSpotAngle);
    mLightUniform->setInnerSpotLightAngle(light->mInnerSpotAngle);
    mLightUniform->setShadowStrength(light->mShadowStrength);
    mLightUniform->setShadowBias(light->mShadowBias);

    if (light->mLightType == LightType::Directional)
    {
        for (int i = 0; i < 5; i++)
        {
            mLightUniform->setDirLightCascadeProj(i, mCascadeOrthoProj[i]);

            glm::vec4 cascadeEnd = glm::vec4(0.0f, 0.0f, mCascadeEnds[i + 1], 1.0f);
            glm::vec4 clipSpaceCascadeEnd = camera->getProjMatrix() * cascadeEnd;

            mLightUniform->setDirLightCascadeEnd(i, clipSpaceCascadeEnd.z);
            mLightUniform->setDirLightCascadeView(i, mCascadeLightView[i]);
        }
    }
    else if (light->mLightType == LightType::Spot)
    {
        mLightUniform->setDirLightCascadeProj(0, mShadowProjMatrix);
        mLightUniform->setDirLightCascadeView(0, mShadowViewMatrix);
    }

    mLightUniform->copyToUniformsToDevice();

    // Configure shader variant
    int64_t variant = static_cast<int64_t>(ShaderMacro::None);

    switch (light->mLightType)
    {
    case LightType::Directional:
        variant = static_cast<int64_t>(ShaderMacro::Directional);
        if (camera->mShadowCascades != ShadowCascades::NoCascades)
        {
            if (camera->mColorTarget == ColorTarget::ShadowCascades)
            {
                variant |= static_cast<int64_t>(ShaderMacro::ShowCascades);
            }
        }
        break;
    case LightType::Spot:
        variant = static_cast<int64_t>(ShaderMacro::Spot);
        break;
    case LightType::Point:
        variant = static_cast<int64_t>(ShaderMacro::Point);
        break;
    }

    switch (light->mShadowType)
    {
    case ShadowType::Hard:
        variant |= static_cast<int64_t>(ShaderMacro::HardShadows);
        break;
    case ShadowType::Soft:
        variant |= static_cast<int64_t>(ShaderMacro::SoftShadows);
        break;
    }

    // Select framebuffer
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

    // Draw to framebuffer
    framebuffer->bind();
    framebuffer->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                             camera->getViewport().mHeight);

    //Shader *shader = nullptr;
    //Material *material = nullptr;

    int modelIndex = 0;
    for (size_t i = 0; i < commands.size(); i++)
    {
        uint16_t materialIndex = commands[i].getMaterialIndex();
        uint16_t meshIndex = commands[i].getMeshIndex();
        uint8_t subMesh = commands[i].getSubMesh();
        bool instanced = commands[i].isInstanced();
        bool indexed = commands[i].isIndexed();

        Material *material = mWorld->getAssetByIndex<Material>(materialIndex);
        Mesh *mesh = mWorld->getAssetByIndex<Mesh>(meshIndex);

        int subMeshVertexStartIndex = mesh->getSubMeshStartIndex(subMesh);
        int subMeshVertexEndIndex = mesh->getSubMeshEndIndex(subMesh);

        Shader *shader = mWorld->getAssetByIndex<Shader>(mWorld->getIndexOf(material->getShaderGuid()));
           
        //if (shader != commands[i].shader)
        //{
        //    shader = commands[i].shader;
        //}

        if (instanced)
        {
            shader->bind(variant | static_cast<int64_t>(ShaderMacro::Instancing));
        }
        else
        {
            shader->bind(shader->getProgramFromVariant(variant) != nullptr ? variant : 0);
        }

        if (light->mShadowType != ShadowType::None)
        {
            if (light->mLightType == LightType::Directional)
            {
                std::vector<void *> tex = {light->getNativeGraphicsShadowCascadeFBO(0)->getDepthTex()->getTexture(),
                                           light->getNativeGraphicsShadowCascadeFBO(1)->getDepthTex()->getTexture(),
                                           light->getNativeGraphicsShadowCascadeFBO(2)->getDepthTex()->getTexture(),
                                           light->getNativeGraphicsShadowCascadeFBO(3)->getDepthTex()->getTexture(),
                                           light->getNativeGraphicsShadowCascadeFBO(4)->getDepthTex()->getTexture()};
                std::vector<int> texUnit = {3, 4, 5, 6, 7};
                shader->setTexture2Ds(Shader::SHADOW_MAP_UNIFORM_ID, texUnit, 5, tex);
            }
            else if (light->mLightType == LightType::Spot)
            {
                shader->setTexture2D(Shader::SHADOW_MAP_UNIFORM_ID, 3,
                                     light->getNativeGrpahicsShadowSpotlightDepthTex()->getTexture());
            }
        }

        material->apply();

        if (instanced)
        {
            VertexBuffer *instanceModelBuffer = mesh->getNativeGraphicsInstanceModelBuffer();

            instanceModelBuffer->bind();
            instanceModelBuffer->setData(models.data() + modelIndex, 0,
                                         sizeof(glm::mat4) * Renderer::INSTANCE_BATCH_SIZE);
            instanceModelBuffer->unbind();
            modelIndex += Renderer::INSTANCE_BATCH_SIZE;
        }
        else
        {
            shader->setMat4(Shader::MODEL_UNIFORM_ID, models[modelIndex]);
            modelIndex++;
        }

        //if (commands[i].meshRendererIndex >= 0)
        //{
        //occlusionQuery.beginQuery(i);
        //}

        if (instanced)
        {
            Renderer::getRenderer()->drawIndexedInstanced(mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
                                                          (subMeshVertexEndIndex - subMeshVertexStartIndex),
                                                          Renderer::INSTANCE_BATCH_SIZE, camera->mQuery);
        }
        else
        {
            if (indexed)
            {
                Renderer::getRenderer()->drawIndexed(mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
                                                     (subMeshVertexEndIndex - subMeshVertexStartIndex), camera->mQuery);
            }
            else
            {
                Renderer::getRenderer()->draw(mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
                                              (subMeshVertexEndIndex - subMeshVertexStartIndex), camera->mQuery);
            }
        }

        //if (commands[i].meshRendererIndex >= 0)
        //{
        //occlusionQuery.endQuery(i);
        //}
    }

    framebuffer->unbind();
}

void ForwardRenderer::renderColorPicking(Camera *camera, const std::vector<DrawCallCommand> &commands,
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
        bool indexed = commands[i].isIndexed();

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

            instanceModelBuffer->bind();
            instanceModelBuffer->setData(models.data() + modelIndex, 0,
                                         sizeof(glm::mat4) * Renderer::INSTANCE_BATCH_SIZE);
            instanceModelBuffer->unbind();

            instanceColorBuffer->bind();
            instanceColorBuffer->setData(colors.data(), 0, sizeof(glm::uvec4) * Renderer::INSTANCE_BATCH_SIZE);
            instanceColorBuffer->unbind();
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

            if (indexed)
            {
                Renderer::getRenderer()->drawIndexed(mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
                                                     (subMeshVertexEndIndex - subMeshVertexStartIndex), camera->mQuery);
            }
            else
            {
                Renderer::getRenderer()->draw(mesh->getNativeGraphicsHandle(), subMeshVertexStartIndex,
                                              (subMeshVertexEndIndex - subMeshVertexStartIndex), camera->mQuery);
            }

            color++;
            modelIndex++;
        }
    }

    camera->getNativeGraphicsColorPickingFBO()->unbind();
}

void ForwardRenderer::renderTransparents()
{
}

void ForwardRenderer::postProcessing()
{
}

void ForwardRenderer::endFrame(Camera *camera)
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

    Renderer::getRenderer()->turnOff(Capability::BackfaceCulling);
}

void ForwardRenderer::calcCascadeOrthoProj(Camera *camera, glm::vec3 lightDirection)
{
    glm::mat4 viewInv = camera->getInvViewMatrix();
    float fov = camera->getFrustum().mFov;
    float aspect = camera->getFrustum().mAspectRatio;
    float tanHalfHFOV = glm::tan(glm::radians(0.5f * fov));
    float tanHalfVFOV = aspect * glm::tan(glm::radians(0.5f * fov));

    for (int i = 0; i < 5; i++)
    {
        float xn = -1.0f * mCascadeEnds[i] * tanHalfHFOV;
        float xf = -1.0f * mCascadeEnds[i + 1] * tanHalfHFOV;
        float yn = -1.0f * mCascadeEnds[i] * tanHalfVFOV;
        float yf = -1.0f * mCascadeEnds[i + 1] * tanHalfVFOV;

        // Find cascade frustum corners
        glm::vec4 frustumCorners[8];
        frustumCorners[0] = glm::vec4(xn, yn, mCascadeEnds[i], 1.0f);
        frustumCorners[1] = glm::vec4(-xn, yn, mCascadeEnds[i], 1.0f);
        frustumCorners[2] = glm::vec4(xn, -yn, mCascadeEnds[i], 1.0f);
        frustumCorners[3] = glm::vec4(-xn, -yn, mCascadeEnds[i], 1.0f);

        frustumCorners[4] = glm::vec4(xf, yf, mCascadeEnds[i + 1], 1.0f);
        frustumCorners[5] = glm::vec4(-xf, yf, mCascadeEnds[i + 1], 1.0f);
        frustumCorners[6] = glm::vec4(xf, -yf, mCascadeEnds[i + 1], 1.0f);
        frustumCorners[7] = glm::vec4(-xf, -yf, mCascadeEnds[i + 1], 1.0f);

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
        float d = std::max(80.0f, -(mCascadeEnds[i + 1] - mCascadeEnds[i]));
        glm::vec3 p = glm::vec3(frustrumCentreWorldSpace.x - d * lightDirection.x,
                                frustrumCentreWorldSpace.y - d * lightDirection.y,
                                frustrumCentreWorldSpace.z - d * lightDirection.z);

        mCascadeLightView[i] = glm::lookAt(p, glm::vec3(frustrumCentreWorldSpace), glm::vec3(0.0f, 1.0f, 0.0f));

        float minX = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::lowest();
        float minY = std::numeric_limits<float>::max();
        float maxY = std::numeric_limits<float>::lowest();
        float minZ = std::numeric_limits<float>::max();
        float maxZ = std::numeric_limits<float>::lowest();

        glm::mat4 cascadeLightView = mCascadeLightView[i];

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
        mCascadeOrthoProj[i] = glm::ortho(minX, maxX, minY, maxY, 0.0f, -minZ);
    }
}