#include "../../include/graphics/GizmoRenderer.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

GizmoRenderer::GizmoRenderer()
{
}

GizmoRenderer::~GizmoRenderer()
{
    destroyGizmoRenderer(mState);
}

void GizmoRenderer::init(World *world)
{
    mWorld = world;

    initializeGizmoRenderer(mWorld, mState);
}

void GizmoRenderer::update(Camera *camera)
{
    for (size_t i = 0; i < mWorld->mBoundingSpheres.size(); i++)
    {
        /*addToDrawList(mWorld->mBoundingSpheres[i], mWorld->mRenderObjects[i].culled ? Color::blue : Color::red);*/
        addToDrawList(mWorld->mBoundingSpheres[i], Color(0.0f, 0.0f, 1.0f, 0.3f) /*Color::blue*/);
    }

    renderLineGizmos(mWorld, camera, mState, mLines);
    renderPlaneGizmos(mWorld, camera, mState, mPlanes);
    renderAABBGizmos(mWorld, camera, mState, mAABBs);
    renderSphereGizmos(mWorld, camera, mState, mSpheres);
    renderFrustumGizmos(mWorld, camera, mState, mFrustums);
}

void GizmoRenderer::drawGrid(Camera* camera)
{
    renderGridGizmo(mWorld, camera, mState);
}

void GizmoRenderer::addToDrawList(const Line &line, const Color &color)
{
    mLines.push_back(LineGizmo(line, color));
}

void GizmoRenderer::addToDrawList(const Ray &ray, float t, const Color &color)
{
    Line line;
    line.mStart = ray.mOrigin;
    line.mEnd = ray.mOrigin + t * ray.mDirection;

    mLines.push_back(LineGizmo(line, color));
}

void GizmoRenderer::addToDrawList(const Sphere& sphere, const Color& color)
{
    mSpheres.push_back(SphereGizmo(sphere, color));
}

void GizmoRenderer::addToDrawList(const AABB &aabb, const Color &color, bool wireframe)
{
    mAABBs.push_back(AABBGizmo(aabb, color, wireframe));
}

void GizmoRenderer::addToDrawList(const Frustum &frustum, const Color &color, bool wireframe)
{
    mFrustums.push_back(FrustumGizmo(frustum, color, wireframe));
}

void GizmoRenderer::addToDrawList(const Plane &plane, const glm::vec3 &extents, const Color &color, bool wireframe)
{
    mPlanes.push_back(PlaneGizmo(plane, extents, color, wireframe));
}

void GizmoRenderer::clearDrawList()
{
    mLines.clear();
    mAABBs.clear();
    mSpheres.clear();
    mFrustums.clear();
    mPlanes.clear();
}


void PhysicsEngine::initializeGizmoRenderer(World *world, GizmoRendererState &state)
{
    Graphics::compileLineShader(state);
    Graphics::compileGizmoShader(state);
    Graphics::compileGridShader(state);

    state.mGridColor = Color(1.0f, 1.0f, 1.0f, 1.0f);

    state.mFrustumVertices.resize(108, 0.0f);
    state.mFrustumNormals.resize(108, 0.0f);

    Graphics::createFrustum(state.mFrustumVertices, state.mFrustumNormals, &state.mFrustumVAO, &state.mFrustumVBO[0],
                            &state.mFrustumVBO[1]);

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

    Graphics::createGrid(state.mGridVertices, &state.mGridVAO, &state.mGridVBO);
}

void PhysicsEngine::destroyGizmoRenderer(GizmoRendererState &state)
{
    Graphics::destroyFrustum(&state.mFrustumVAO, &state.mFrustumVBO[0], &state.mFrustumVBO[1]);
    Graphics::destroyGrid(&state.mGridVAO, &state.mGridVBO);
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
        colors[8 * i + 0] = gizmos[i].mColor.mR;
        colors[8 * i + 1] = gizmos[i].mColor.mG;
        colors[8 * i + 2] = gizmos[i].mColor.mB;
        colors[8 * i + 3] = gizmos[i].mColor.mA;

        colors[8 * i + 4] = gizmos[i].mColor.mR;
        colors[8 * i + 5] = gizmos[i].mColor.mG;
        colors[8 * i + 6] = gizmos[i].mColor.mB;
        colors[8 * i + 7] = gizmos[i].mColor.mA;
    }

    unsigned int lineVAO;
    unsigned int lineVBO[2];

    Graphics::createLine(vertices, colors, &lineVAO, &lineVBO[0], &lineVBO[1]);

    Graphics::bindFramebuffer(camera->getNativeGraphicsMainFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    glm::mat4 mvp = camera->getProjMatrix() * camera->getViewMatrix();

    Graphics::use(state.mLineShaderProgram);
    Graphics::setMat4(state.mLineShaderMVPLoc, mvp);

    Graphics::renderLines(lineVAO, 0, (int)vertices.size() / 3);

    Graphics::unbindFramebuffer();

    Graphics::destroyLine(&lineVAO, &lineVBO[0], &lineVBO[1]);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderSphereGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                       const std::vector<SphereGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    Graphics::turnOn(Capability::Blending);
    Graphics::setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    Transform *transform = camera->getComponent<Transform>();

    Mesh *mesh = world->getPrimtiveMesh(PrimitiveType::Sphere);

    Graphics::bindFramebuffer(camera->getNativeGraphicsMainFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Graphics::bindVertexArray(mesh->getNativeGraphicsVAO());

    Graphics::use(state.mGizmoShaderProgram);

    Graphics::setVec3(state.mGizmoShaderLightPosLoc, transform->getPosition());
    Graphics::setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    Graphics::setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());

    for (size_t i = 0; i < gizmos.size(); i++)
    {
        glm::mat4 model = glm::translate(glm::mat4(1.0f), gizmos[i].mSphere.mCentre);
        model = glm::scale(model,
                           glm::vec3(gizmos[i].mSphere.mRadius, gizmos[i].mSphere.mRadius, gizmos[i].mSphere.mRadius));

        Graphics::setColor(state.mGizmoShaderColorLoc, gizmos[i].mColor);
        Graphics::setMat4(state.mGizmoShaderModelLoc, model);

        Graphics::renderWithCurrentlyBoundVAO(0, (int)mesh->getVertices().size() / 3);
    }

    Graphics::unbindVertexArray();
    Graphics::unbindFramebuffer();

    Graphics::turnOff(Capability::Blending);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderAABBGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                     const std::vector<AABBGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    Graphics::turnOn(Capability::Blending);
    Graphics::setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    Transform *transform = camera->getComponent<Transform>();

    Mesh *mesh = world->getPrimtiveMesh(PrimitiveType::Cube);

    Graphics::bindFramebuffer(camera->getNativeGraphicsMainFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);
 
    Graphics::bindVertexArray(mesh->getNativeGraphicsVAO());

    Graphics::use(state.mGizmoShaderProgram);

    Graphics::setVec3(state.mGizmoShaderLightPosLoc, transform->getPosition());
    Graphics::setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    Graphics::setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());

    for (size_t i = 0; i < gizmos.size(); i++)
    {
        glm::mat4 model = glm::translate(glm::mat4(1.0f), gizmos[i].mAABB.mCentre);
        model = glm::scale(model, gizmos[i].mAABB.mSize);

        Graphics::setColor(state.mGizmoShaderColorLoc, gizmos[i].mColor);
        Graphics::setMat4(state.mGizmoShaderModelLoc, model);

        Graphics::renderWithCurrentlyBoundVAO(0, (int)mesh->getVertices().size() / 3);
    }

    Graphics::unbindVertexArray();   
    Graphics::unbindFramebuffer();

    Graphics::turnOff(Capability::Blending);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderPlaneGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                      const std::vector<PlaneGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    Graphics::turnOn(Capability::Blending);
    Graphics::setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    Transform *transform = camera->getComponent<Transform>();

    Mesh *mesh = world->getPrimtiveMesh(PrimitiveType::Plane);

    Graphics::bindFramebuffer(camera->getNativeGraphicsMainFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Graphics::bindVertexArray(mesh->getNativeGraphicsVAO());

    Graphics::use(state.mGizmoShaderProgram);

    Graphics::setVec3(state.mGizmoShaderLightPosLoc, transform->getPosition());
    Graphics::setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    Graphics::setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());

    for (size_t i = 0; i < gizmos.size(); i++)
    {
        glm::mat4 model = glm::translate(glm::mat4(1.0f), gizmos[i].mPlane.mX0);
        glm::vec3 a = glm::vec3(0, 0, 1);
        glm::vec3 b = gizmos[i].mPlane.mNormal;
        float d = glm::dot(a, b);
        glm::vec3 c = glm::cross(a, b);
        float angle = glm::atan(glm::length(c), d);

        model = glm::rotate(model, angle, c);

        Graphics::setColor(state.mGizmoShaderColorLoc, gizmos[i].mColor);
        Graphics::setMat4(state.mGizmoShaderModelLoc, model);

        Graphics::renderWithCurrentlyBoundVAO(0, (int)mesh->getVertices().size() / 3);
    }

    Graphics::unbindVertexArray();    
    Graphics::unbindFramebuffer();

    Graphics::turnOff(Capability::Blending);

    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderShadedFrustumGizmo(World *world, Camera *camera, GizmoRendererState &state,
                                             const FrustumGizmo &gizmo)
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

    Transform *transform = camera->getComponent<Transform>();

    Graphics::use(state.mGizmoShaderProgram);
    Graphics::setVec3(state.mGizmoShaderLightPosLoc, transform->getPosition());
    Graphics::setMat4(state.mGizmoShaderModelLoc, glm::mat4(1.0f));
    Graphics::setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    Graphics::setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());
    Graphics::setColor(state.mGizmoShaderColorLoc, gizmo.mColor);

    Graphics::updateFrustum(state.mFrustumVertices, state.mFrustumNormals, state.mFrustumVBO[0],
                            state.mFrustumVBO[1]);

    Graphics::renderWithCurrentlyBoundVAO(0, (int)state.mFrustumVertices.size() / 3);
}

void PhysicsEngine::renderWireframeFrustumGizmo(World *world, Camera *camera, GizmoRendererState &state,
                                                const FrustumGizmo &gizmo)
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

    Graphics::use(state.mLineShaderProgram);
    Graphics::setMat4(state.mLineShaderMVPLoc, mvp);

    Graphics::updateFrustum(state.mFrustumVertices, state.mFrustumVBO[0]);
    
    Graphics::renderLinesWithCurrentlyBoundVAO(0, 24);
}

void PhysicsEngine::renderFrustumGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                        const std::vector<FrustumGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    Graphics::turnOn(Capability::Blending);
    Graphics::setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    Graphics::bindFramebuffer(camera->getNativeGraphicsMainFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Graphics::bindVertexArray(state.mFrustumVAO);

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

    Graphics::unbindVertexArray();
    Graphics::unbindFramebuffer();

    Graphics::turnOff(Capability::Blending);
    Graphics::checkError(__LINE__, __FILE__);
}

void PhysicsEngine::renderGridGizmo(World *world, Camera *camera, GizmoRendererState &state)
{
    Graphics::checkError(__LINE__, __FILE__);
    Graphics::turnOn(Capability::Blending);
    Graphics::turnOn(Capability::LineSmoothing);
    Graphics::setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    Graphics::checkError(__LINE__, __FILE__);
    Graphics::bindFramebuffer(camera->getNativeGraphicsMainFBO());
    Graphics::setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Graphics::checkError(__LINE__, __FILE__);
    glm::mat4 mvp = camera->getProjMatrix() * camera->getViewMatrix();

    Graphics::use(state.mGridShaderProgram);
    Graphics::setMat4(state.mGridShaderMVPLoc, mvp);
    Graphics::setColor(state.mGridShaderColorLoc, state.mGridColor);

    Graphics::checkError(__LINE__, __FILE__);

    Graphics::renderLines(0, (int)state.mGridVertices.size(), state.mGridVAO);

    Graphics::checkError(__LINE__, __FILE__);
    Graphics::unbindFramebuffer();

    Graphics::turnOff(Capability::Blending);
    Graphics::turnOff(Capability::LineSmoothing);

    Graphics::checkError(__LINE__, __FILE__);
}