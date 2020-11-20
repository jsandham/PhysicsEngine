#include "../../include/graphics/GizmoRendererPasses.h"
#include "../../include/graphics/Graphics.h"

#include "../../include/glm/gtx/rotate_vector.hpp"

using namespace PhysicsEngine;

void PhysicsEngine::initializeGizmoRenderer(World *world, GizmoRendererState &state)
{
    state.mLineShader = world->getAssetById<Shader>(world->getLineShaderId());
    state.mGizmoShader = world->getAssetById<Shader>(world->getGizmoShaderId());

    assert(state.mLineShader != NULL);
    assert(state.mGizmoShader != NULL);

    state.mLineShader->compile();
    state.mGizmoShader->compile();

    // cache internal shader uniforms
    state.mLineShaderProgram = state.mLineShader->getProgramFromVariant(ShaderVariant::None);
    state.mLineShaderMVPLoc = state.mLineShader->findUniformLocation("mvp", state.mLineShaderProgram);

    state.mGizmoShaderProgram = state.mGizmoShader->getProgramFromVariant(ShaderVariant::None);
    state.mGizmoShaderColorLoc = state.mGizmoShader->findUniformLocation("color", state.mGizmoShaderProgram);
    state.mGizmoShaderLightPosLoc = state.mGizmoShader->findUniformLocation("lightPos", state.mGizmoShaderProgram);
    state.mGizmoShaderModelLoc = state.mGizmoShader->findUniformLocation("model", state.mGizmoShaderProgram);
    state.mGizmoShaderViewLoc = state.mGizmoShader->findUniformLocation("view", state.mGizmoShaderProgram);
    state.mGizmoShaderProjLoc = state.mGizmoShader->findUniformLocation("projection", state.mGizmoShaderProgram);
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

void PhysicsEngine::renderAABBGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                     const std::vector<AABBGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Transform *transform = camera->getComponent<Transform>(world);

    Mesh *mesh = world->getAssetById<Mesh>(world->getCubeMesh());

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

    Transform *transform = camera->getComponent<Transform>(world);

    Mesh *mesh = world->getAssetById<Mesh>(world->getPlaneMesh());

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

void PhysicsEngine::renderSphereGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                       const std::vector<SphereGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Transform *transform = camera->getComponent<Transform>(world);

    Mesh *mesh = world->getAssetById<Mesh>(world->getSphereMesh());

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

void PhysicsEngine::renderFrustumGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                        const std::vector<FrustumGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Transform *transform = camera->getComponent<Transform>(world);

    std::vector<float> vertices(108, 0.0f);
    std::vector<float> normals(108, 0.0f);

    GLuint frustumVAO;
    GLuint frustumVBO[2];

    glGenVertexArrays(1, &frustumVAO);
    glBindVertexArray(frustumVAO);

    glGenBuffers(2, &frustumVBO[0]);
    glBindBuffer(GL_ARRAY_BUFFER, frustumVBO[0]);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

    glGenBuffers(1, &frustumVBO[1]);
    glBindBuffer(GL_ARRAY_BUFFER, frustumVBO[1]);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), &normals[0], GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());
    glViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
               camera->getViewport().mHeight);

    glBindVertexArray(frustumVAO);

    state.mGizmoShader->use(state.mGizmoShaderProgram);
    state.mGizmoShader->setVec3(state.mGizmoShaderLightPosLoc, transform->mPosition);
    state.mGizmoShader->setMat4(state.mGizmoShaderModelLoc, glm::mat4());
    state.mGizmoShader->setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    state.mGizmoShader->setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());

    for (size_t i = 0; i < gizmos.size(); i++)
    {
        glm::vec3 temp[36];
        temp[0] = gizmos[i].mFrustum.mNtl;
        temp[1] = gizmos[i].mFrustum.mNtr;
        temp[2] = gizmos[i].mFrustum.mNbr;

        temp[3] = gizmos[i].mFrustum.mNtl;
        temp[4] = gizmos[i].mFrustum.mNbr;
        temp[5] = gizmos[i].mFrustum.mNbl;

        temp[6] = gizmos[i].mFrustum.mFtl;
        temp[7] = gizmos[i].mFrustum.mFtr;
        temp[8] = gizmos[i].mFrustum.mFbr;

        temp[9] = gizmos[i].mFrustum.mFtl;
        temp[10] = gizmos[i].mFrustum.mFbr;
        temp[11] = gizmos[i].mFrustum.mFbl;

        temp[12] = gizmos[i].mFrustum.mNtl;
        temp[13] = gizmos[i].mFrustum.mFtl;
        temp[14] = gizmos[i].mFrustum.mFtr;

        temp[15] = gizmos[i].mFrustum.mNtl;
        temp[16] = gizmos[i].mFrustum.mFtr;
        temp[17] = gizmos[i].mFrustum.mNtr;

        temp[18] = gizmos[i].mFrustum.mNbl;
        temp[19] = gizmos[i].mFrustum.mFbl;
        temp[20] = gizmos[i].mFrustum.mFbr;

        temp[21] = gizmos[i].mFrustum.mNbl;
        temp[22] = gizmos[i].mFrustum.mFbr;
        temp[23] = gizmos[i].mFrustum.mNbr;

        temp[24] = gizmos[i].mFrustum.mNbl;
        temp[25] = gizmos[i].mFrustum.mFbl;
        temp[26] = gizmos[i].mFrustum.mFtl;

        temp[27] = gizmos[i].mFrustum.mNbl;
        temp[28] = gizmos[i].mFrustum.mFtl;
        temp[29] = gizmos[i].mFrustum.mNtl;

        temp[30] = gizmos[i].mFrustum.mNbr;
        temp[31] = gizmos[i].mFrustum.mFbr;
        temp[32] = gizmos[i].mFrustum.mFtr;

        temp[33] = gizmos[i].mFrustum.mNbr;
        temp[34] = gizmos[i].mFrustum.mFtr;
        temp[35] = gizmos[i].mFrustum.mNtr;

        for (int j = 0; j < 36; j++)
        {
            vertices[3 * j + 0] = temp[j].x;
            vertices[3 * j + 1] = temp[j].y;
            vertices[3 * j + 2] = temp[j].z;
        }

        for (int j = 0; j < 6; j++)
        {
            glm::vec3 a = temp[6 * j + 1] - temp[6 * j];
            glm::vec3 b = temp[6 * j + 2] - temp[6 * j];

            glm::vec3 normal = glm::cross(a, b);

            for (int k = 0; k < 6; k++)
            {
                normals[18 * j + 3 * k + 0] = normal.x;
                normals[18 * j + 3 * k + 1] = normal.y;
                normals[18 * j + 3 * k + 2] = normal.z;
            }
        }

        glBindBuffer(GL_ARRAY_BUFFER, frustumVBO[0]);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), &vertices[0]);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindBuffer(GL_ARRAY_BUFFER, frustumVBO[1]);
        glBufferSubData(GL_ARRAY_BUFFER, 0, normals.size() * sizeof(float), &normals[0]);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        state.mGizmoShader->setColor(state.mGizmoShaderColorLoc, gizmos[i].mColor);

        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(vertices.size() / 3));
    }

    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDeleteVertexArrays(1, &frustumVAO);
    glDeleteBuffers(2, &frustumVBO[0]);

    glDisable(GL_BLEND);

    Graphics::checkError(__LINE__, __FILE__);
}