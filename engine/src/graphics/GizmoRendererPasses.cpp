#include "../../include/graphics/GizmoRendererPasses.h"
#include "../../include/graphics/Graphics.h"

#include "glm/gtx/rotate_vector.hpp"

using namespace PhysicsEngine;

void PhysicsEngine::initializeGizmoRenderer(World *world, GizmoRendererState &state)
{
    state.mLineShader = world->getAssetById<Shader>(world->getAssetId("data\\shaders\\opengl\\line.shader"));
    state.mGizmoShader = world->getAssetById<Shader>(world->getAssetId("data\\shaders\\opengl\\gizmo.shader"));
    state.mGridShader = world->getAssetById<Shader>(world->getAssetId("data\\shaders\\opengl\\grid.shader"));

    assert(state.mLineShader != NULL);
    assert(state.mGizmoShader != NULL);
    assert(state.mGridShader != NULL);

    state.mLineShader->compile();
    state.mGizmoShader->compile();
    state.mGridShader->compile();

    // cache internal shader uniforms
    state.mLineShaderProgram = state.mLineShader->getProgramFromVariant(static_cast<int64_t>(ShaderMacro::None));
    state.mLineShaderMVPLoc = state.mLineShader->findUniformLocation("mvp", state.mLineShaderProgram);

    state.mGizmoShaderProgram = state.mGizmoShader->getProgramFromVariant(static_cast<int64_t>(ShaderMacro::None));
    state.mGizmoShaderColorLoc = state.mGizmoShader->findUniformLocation("color", state.mGizmoShaderProgram);
    state.mGizmoShaderLightPosLoc = state.mGizmoShader->findUniformLocation("lightPos", state.mGizmoShaderProgram);
    state.mGizmoShaderModelLoc = state.mGizmoShader->findUniformLocation("model", state.mGizmoShaderProgram);
    state.mGizmoShaderViewLoc = state.mGizmoShader->findUniformLocation("view", state.mGizmoShaderProgram);
    state.mGizmoShaderProjLoc = state.mGizmoShader->findUniformLocation("projection", state.mGizmoShaderProgram);

    state.mGridShaderProgram = state.mGridShader->getProgramFromVariant(static_cast<int64_t>(ShaderMacro::None));
    state.mGridShaderMVPLoc = state.mGridShader->findUniformLocation("mvp", state.mGridShaderProgram);
    state.mGridShaderColorLoc = state.mGridShader->findUniformLocation("color", state.mGridShaderProgram);

    state.mGridColor = Color(1.0f, 1.0f, 1.0f, 1.0f);

    state.mFrustumVertices.resize(108, 0.0f);
    state.mFrustumNormals.resize(108, 0.0f);

    glGenVertexArrays(1, &state.mFrustumVAO);
    glBindVertexArray(state.mFrustumVAO);

    glGenBuffers(2, &state.mFrustumVBO[0]);
    glBindBuffer(GL_ARRAY_BUFFER, state.mFrustumVBO[0]);
    glBufferData(GL_ARRAY_BUFFER, state.mFrustumVertices.size() * sizeof(float), &state.mFrustumVertices[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glGenBuffers(1, &state.mFrustumVBO[1]);
    glBindBuffer(GL_ARRAY_BUFFER, state.mFrustumVBO[1]);
    glBufferData(GL_ARRAY_BUFFER, state.mFrustumNormals.size() * sizeof(float), &state.mFrustumNormals[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    for (int i = -100; i < 100; i++)
    {
        glm::vec3 start = glm::vec3(i, 0.0f, -100.0f);
        glm::vec3 end = glm::vec3(i, 0.0f, 100.0f);

        state.mGridVertices.push_back(start);
        state.mGridVertices.push_back(end);
    }

    for (int i = -100; i < 100; i++)
    {
        glm::vec3 start = glm::vec3(-100.0f, 0.0f, i);
        glm::vec3 end = glm::vec3(100.0f, 0.0f, i);

        state.mGridVertices.push_back(start);
        state.mGridVertices.push_back(end);
    }

    glGenVertexArrays(1, &state.mGridVAO);
    glBindVertexArray(state.mGridVAO);

    glGenBuffers(1, &state.mGridVBO);
    glBindBuffer(GL_ARRAY_BUFFER, state.mGridVBO);
    glBufferData(GL_ARRAY_BUFFER, state.mGridVertices.size() * sizeof(glm::vec3), &state.mGridVertices[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void PhysicsEngine::destroyGizmoRenderer(GizmoRendererState& state)
{
    glDeleteVertexArrays(1, &state.mFrustumVAO);
    glDeleteBuffers(2, &state.mFrustumVBO[0]);

    glDeleteVertexArrays(1, &state.mGridVAO);
    glDeleteBuffers(1, &state.mGridVBO);
}

void PhysicsEngine::renderLineGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                     const std::vector<LineGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    std::vector<float> vertices(6 * gizmos.size());

    for (size_t i = 0; i < gizmos.size(); i++)
    {
        vertices[6 * i + 0] = gizmos[i].mLine.mStart.x;
        vertices[6 * i + 1] = gizmos[i].mLine.mStart.y;
        vertices[6 * i + 2] = gizmos[i].mLine.mStart.z;

        vertices[6 * i + 3] = gizmos[i].mLine.mEnd.x;
        vertices[6 * i + 4] = gizmos[i].mLine.mEnd.y;
        vertices[6 * i + 5] = gizmos[i].mLine.mEnd.z;
    }

    std::vector<float> colors(8 * gizmos.size());

    for (size_t i = 0; i < gizmos.size(); i++)
    {
        colors[8 * i + 0] = gizmos[i].mColor.r;
        colors[8 * i + 1] = gizmos[i].mColor.g;
        colors[8 * i + 2] = gizmos[i].mColor.b;
        colors[8 * i + 3] = gizmos[i].mColor.a;

        colors[8 * i + 4] = gizmos[i].mColor.r;
        colors[8 * i + 5] = gizmos[i].mColor.g;
        colors[8 * i + 6] = gizmos[i].mColor.b;
        colors[8 * i + 7] = gizmos[i].mColor.a;
    }

    GLuint lineVAO;
    GLuint lineVBO[2];

    glGenVertexArrays(1, &lineVAO);
    glBindVertexArray(lineVAO);

    glGenBuffers(2, &lineVBO[0]);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO[0]);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    glGenBuffers(1, &lineVBO[1]);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO[1]);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float), &colors[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GL_FLOAT), 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());
    glViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
               camera->getViewport().mHeight);

    glm::mat4 mvp = camera->getProjMatrix() * camera->getViewMatrix();

    state.mLineShader->use(state.mLineShaderProgram);
    state.mLineShader->setMat4(state.mLineShaderMVPLoc, mvp);

    glBindVertexArray(lineVAO);
    glDrawArrays(GL_LINES, 0, (GLsizei)(vertices.size() / 3));
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDeleteVertexArrays(1, &lineVAO);
    glDeleteBuffers(2, &lineVBO[0]);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderSphereGizmos(World* world, Camera* camera, GizmoRendererState& state,
    const std::vector<SphereGizmo>& gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Transform* transform = camera->getComponent<Transform>();

    Mesh* mesh = world->getAssetById<Mesh>(world->getAssetId("data\\meshes\\sphere.mesh"));

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());
    glViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
        camera->getViewport().mHeight);

    glBindVertexArray(mesh->getNativeGraphicsVAO());

    state.mGizmoShader->use(state.mGizmoShaderProgram);

    state.mGizmoShader->setVec3(state.mGizmoShaderLightPosLoc, transform->mPosition);
    state.mGizmoShader->setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    state.mGizmoShader->setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());

    for (size_t i = 0; i < gizmos.size(); i++)
    {
        glm::mat4 model = glm::translate(glm::mat4(), gizmos[i].mSphere.mCentre);
        model = glm::scale(model,
            glm::vec3(gizmos[i].mSphere.mRadius, gizmos[i].mSphere.mRadius, gizmos[i].mSphere.mRadius));

        state.mGizmoShader->setColor(state.mGizmoShaderColorLoc, gizmos[i].mColor);
        state.mGizmoShader->setMat4(state.mGizmoShaderModelLoc, model);

        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(mesh->getVertices().size() / 3));
    }

    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDisable(GL_BLEND);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderAABBGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                     const std::vector<AABBGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Transform *transform = camera->getComponent<Transform>();

    Mesh *mesh = world->getAssetById<Mesh>(world->getAssetId("data\\meshes\\cube.mesh"));

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());
    glViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
               camera->getViewport().mHeight);

    glBindVertexArray(mesh->getNativeGraphicsVAO());

    state.mGizmoShader->use(state.mGizmoShaderProgram);

    state.mGizmoShader->setVec3(state.mGizmoShaderLightPosLoc, transform->mPosition);
    state.mGizmoShader->setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    state.mGizmoShader->setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());

    for (size_t i = 0; i < gizmos.size(); i++)
    {
        glm::mat4 model = glm::translate(glm::mat4(), gizmos[i].mAABB.mCentre);
        model = glm::scale(model, gizmos[i].mAABB.mSize);

        state.mGizmoShader->setColor(state.mGizmoShaderColorLoc, gizmos[i].mColor);
        state.mGizmoShader->setMat4(state.mGizmoShaderModelLoc, model);

        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(mesh->getVertices().size() / 3));
    }

    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDisable(GL_BLEND);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderPlaneGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                      const std::vector<PlaneGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Transform *transform = camera->getComponent<Transform>();

    Mesh *mesh = world->getAssetById<Mesh>(world->getAssetId("data\\meshes\\plane.mesh"));

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());
    glViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
               camera->getViewport().mHeight);

    glBindVertexArray(mesh->getNativeGraphicsVAO());

    state.mGizmoShader->use(state.mGizmoShaderProgram);

    state.mGizmoShader->setVec3(state.mGizmoShaderLightPosLoc, transform->mPosition);
    state.mGizmoShader->setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    state.mGizmoShader->setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());

    for (size_t i = 0; i < gizmos.size(); i++)
    {
        glm::mat4 model = glm::translate(glm::mat4(), gizmos[i].mPlane.mX0);
        glm::vec3 a = glm::vec3(0, 0, 1);
        glm::vec3 b = gizmos[i].mPlane.mNormal;
        float d = glm::dot(a, b);
        glm::vec3 c = glm::cross(a, b);
        float angle = glm::atan(glm::length(c), d);

        model = glm::rotate(model, angle, c);

        state.mGizmoShader->setColor(state.mGizmoShaderColorLoc, gizmos[i].mColor);
        state.mGizmoShader->setMat4(state.mGizmoShaderModelLoc, model);

        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(mesh->getVertices().size() / 3));
    }

    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDisable(GL_BLEND);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderShadedFrustumGizmo(World* world, Camera* camera, GizmoRendererState& state, const FrustumGizmo& gizmo)
{
    glm::vec3 temp[36];
    temp[0] = gizmo.mFrustum.mNtl;
    temp[1] = gizmo.mFrustum.mNtr;
    temp[2] = gizmo.mFrustum.mNbr;

    temp[3] = gizmo.mFrustum.mNtl;
    temp[4] = gizmo.mFrustum.mNbr;
    temp[5] = gizmo.mFrustum.mNbl;

    temp[6] = gizmo.mFrustum.mFtl;
    temp[7] = gizmo.mFrustum.mFtr;
    temp[8] = gizmo.mFrustum.mFbr;

    temp[9] = gizmo.mFrustum.mFtl;
    temp[10] = gizmo.mFrustum.mFbr;
    temp[11] = gizmo.mFrustum.mFbl;

    temp[12] = gizmo.mFrustum.mNtl;
    temp[13] = gizmo.mFrustum.mFtl;
    temp[14] = gizmo.mFrustum.mFtr;

    temp[15] = gizmo.mFrustum.mNtl;
    temp[16] = gizmo.mFrustum.mFtr;
    temp[17] = gizmo.mFrustum.mNtr;

    temp[18] = gizmo.mFrustum.mNbl;
    temp[19] = gizmo.mFrustum.mFbl;
    temp[20] = gizmo.mFrustum.mFbr;

    temp[21] = gizmo.mFrustum.mNbl;
    temp[22] = gizmo.mFrustum.mFbr;
    temp[23] = gizmo.mFrustum.mNbr;

    temp[24] = gizmo.mFrustum.mNbl;
    temp[25] = gizmo.mFrustum.mFbl;
    temp[26] = gizmo.mFrustum.mFtl;

    temp[27] = gizmo.mFrustum.mNbl;
    temp[28] = gizmo.mFrustum.mFtl;
    temp[29] = gizmo.mFrustum.mNtl;

    temp[30] = gizmo.mFrustum.mNbr;
    temp[31] = gizmo.mFrustum.mFbr;
    temp[32] = gizmo.mFrustum.mFtr;

    temp[33] = gizmo.mFrustum.mNbr;
    temp[34] = gizmo.mFrustum.mFtr;
    temp[35] = gizmo.mFrustum.mNtr;

    for (int j = 0; j < 36; j++)
    {
        state.mFrustumVertices[3 * j + 0] = temp[j].x;
        state.mFrustumVertices[3 * j + 1] = temp[j].y;
        state.mFrustumVertices[3 * j + 2] = temp[j].z;
    }

    for (int j = 0; j < 6; j++)
    {
        glm::vec3 a = temp[6 * j + 1] - temp[6 * j];
        glm::vec3 b = temp[6 * j + 2] - temp[6 * j];

        glm::vec3 normal = glm::cross(a, b);

        for (int k = 0; k < 6; k++)
        {
            state.mFrustumNormals[18 * j + 3 * k + 0] = normal.x;
            state.mFrustumNormals[18 * j + 3 * k + 1] = normal.y;
            state.mFrustumNormals[18 * j + 3 * k + 2] = normal.z;
        }
    }

    Transform* transform = camera->getComponent<Transform>();

    state.mGizmoShader->use(state.mGizmoShaderProgram);
    state.mGizmoShader->setVec3(state.mGizmoShaderLightPosLoc, transform->mPosition);
    state.mGizmoShader->setMat4(state.mGizmoShaderModelLoc, glm::mat4());
    state.mGizmoShader->setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    state.mGizmoShader->setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());
    state.mGizmoShader->setColor(state.mGizmoShaderColorLoc, gizmo.mColor);

    glBindBuffer(GL_ARRAY_BUFFER, state.mFrustumVBO[0]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, state.mFrustumVertices.size() * sizeof(float), &state.mFrustumVertices[0]);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, state.mFrustumVBO[1]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, state.mFrustumNormals.size() * sizeof(float), &state.mFrustumNormals[0]);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(state.mFrustumVertices.size() / 3));
}

void PhysicsEngine::renderWireframeFrustumGizmo(World* world, Camera* camera, GizmoRendererState& state, const FrustumGizmo& gizmo)
{
    glm::vec3 temp[24];
    temp[0] = gizmo.mFrustum.mNtl;
    temp[1] = gizmo.mFrustum.mFtl;
    temp[2] = gizmo.mFrustum.mNtr;
    temp[3] = gizmo.mFrustum.mFtr;
    temp[4] = gizmo.mFrustum.mNbl;
    temp[5] = gizmo.mFrustum.mFbl;
    temp[6] = gizmo.mFrustum.mNbr;
    temp[7] = gizmo.mFrustum.mFbr;

    temp[8] = gizmo.mFrustum.mNtl;
    temp[9] = gizmo.mFrustum.mNtr;
    temp[10] = gizmo.mFrustum.mNtr;
    temp[11] = gizmo.mFrustum.mNbr;
    temp[12] = gizmo.mFrustum.mNbr;
    temp[13] = gizmo.mFrustum.mNbl;
    temp[14] = gizmo.mFrustum.mNbl;
    temp[15] = gizmo.mFrustum.mNtl;

    temp[16] = gizmo.mFrustum.mFtl;
    temp[17] = gizmo.mFrustum.mFtr;
    temp[18] = gizmo.mFrustum.mFtr;
    temp[19] = gizmo.mFrustum.mFbr;
    temp[20] = gizmo.mFrustum.mFbr;
    temp[21] = gizmo.mFrustum.mFbl;
    temp[22] = gizmo.mFrustum.mFbl;
    temp[23] = gizmo.mFrustum.mFtl;

    for (int j = 0; j < 24; j++)
    {
        state.mFrustumVertices[3 * j + 0] = temp[j].x;
        state.mFrustumVertices[3 * j + 1] = temp[j].y;
        state.mFrustumVertices[3 * j + 2] = temp[j].z;
    }

    glm::mat4 mvp = camera->getProjMatrix() * camera->getViewMatrix();

    state.mLineShader->use(state.mLineShaderProgram);
    state.mLineShader->setMat4(state.mLineShaderMVPLoc, mvp);

    glBindBuffer(GL_ARRAY_BUFFER, state.mFrustumVBO[0]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 24 * 3 * sizeof(float), &state.mFrustumVertices[0]);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDrawArrays(GL_LINES, 0, (GLsizei)(24));
}

void PhysicsEngine::renderFrustumGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                        const std::vector<FrustumGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());
    glViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
        camera->getViewport().mHeight);

    glBindVertexArray(state.mFrustumVAO);

    for (size_t i = 0; i < gizmos.size(); i++)
    {
        if (!gizmos[i].mWireFrame)
        {
            renderShadedFrustumGizmo(world, camera, state, gizmos[i]);
        }
        else
        {
            renderWireframeFrustumGizmo(world, camera, state, gizmos[i]);
        }
    }

    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDisable(GL_BLEND);

    Graphics::checkError(__LINE__, __FILE__);
}


void PhysicsEngine::renderGridGizmo(World* world, Camera* camera, GizmoRendererState& state)
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());
    glViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
        camera->getViewport().mHeight);

    glm::mat4 mvp = camera->getProjMatrix() * camera->getViewMatrix();

    state.mGridShader->use(state.mGridShaderProgram);
    state.mGridShader->setMat4(state.mGridShaderMVPLoc, mvp);
    state.mGridShader->setColor(state.mGridShaderColorLoc, state.mGridColor);

    glBindVertexArray(state.mGridVAO);
    glDrawArrays(GL_LINES, 0, (GLsizei)(state.mGridVertices.size()));
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDisable(GL_BLEND);

    Graphics::checkError(__LINE__, __FILE__);
}