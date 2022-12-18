#include "../../include/graphics/GizmoRenderer.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

static void initializeGizmoRenderer(World* world, GizmoRendererState& state);
static void destroyGizmoRenderer(GizmoRendererState& state);
static void renderLineGizmos(World* world, Camera* camera, GizmoRendererState& state, const std::vector<LineGizmo>& gizmos);
static void renderPlaneGizmos(World* world, Camera* camera, GizmoRendererState& state, const std::vector<PlaneGizmo>& gizmos);
static void renderAABBGizmos(World* world, Camera* camera, GizmoRendererState& state, const std::vector<AABBGizmo>& gizmos);
static void renderSphereGizmos(World* world, Camera* camera, GizmoRendererState& state,
    const std::vector<SphereGizmo>& gizmos);
static void renderFrustumGizmos(World* world, Camera* camera, GizmoRendererState& state,
    const std::vector<FrustumGizmo>& gizmos);
static void renderShadedFrustumGizmo(World* world, Camera* camera, GizmoRendererState& state, const FrustumGizmo& gizmo);
static void renderWireframeFrustumGizmo(World* world, Camera* camera, GizmoRendererState& state, const FrustumGizmo& gizmo);
static void renderGridGizmo(World* world, Camera* camera, GizmoRendererState& state);

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


static void initializeGizmoRenderer(World *world, GizmoRendererState &state)
{
    state.mLineShaderProgram = RendererShaders::getLineShader();
    state.mGizmoShaderProgram = RendererShaders::getGizmoShader();
    state.mGridShaderProgram = RendererShaders::getGridShader();

    state.mLineShaderMVPLoc = state.mLineShaderProgram->findUniformLocation("mvp");
    state.mGizmoShaderModelLoc = state.mGizmoShaderProgram->findUniformLocation("model");
    state.mGizmoShaderViewLoc = state.mGizmoShaderProgram->findUniformLocation("view");
    state.mGizmoShaderProjLoc = state.mGizmoShaderProgram->findUniformLocation("projection");
    state.mGizmoShaderColorLoc = state.mGizmoShaderProgram->findUniformLocation("color");
    state.mGizmoShaderLightPosLoc = state.mGizmoShaderProgram->findUniformLocation("lightPos");
    state.mGridShaderMVPLoc = state.mGridShaderProgram->findUniformLocation("mvp");
    state.mGridShaderColorLoc = state.mGridShaderProgram->findUniformLocation("color");

    /*state.mLineShaderProgram = RendererShaders::getRendererShaders()->getLineShader().mProgram;
    state.mLineShaderMVPLoc = RendererShaders::getRendererShaders()->getLineShader().mMVPLoc;

    state.mGizmoShaderProgram = RendererShaders::getRendererShaders()->getGizmoShader().mProgram;
    state.mGizmoShaderModelLoc = RendererShaders::getRendererShaders()->getGizmoShader().mModelLoc;
    state.mGizmoShaderViewLoc = RendererShaders::getRendererShaders()->getGizmoShader().mViewLoc;
    state.mGizmoShaderProjLoc = RendererShaders::getRendererShaders()->getGizmoShader().mProjLoc;
    state.mGizmoShaderColorLoc = RendererShaders::getRendererShaders()->getGizmoShader().mColorLoc;
    state.mGizmoShaderLightPosLoc = RendererShaders::getRendererShaders()->getGizmoShader().mLightPosLoc;

    state.mGridShaderProgram = RendererShaders::getRendererShaders()->getGridShader().mProgram;
    state.mGridShaderMVPLoc = RendererShaders::getRendererShaders()->getGridShader().mMVPLoc;
    state.mGridShaderColorLoc = RendererShaders::getRendererShaders()->getGridShader().mColorLoc;*/

    state.mGridColor = Color(1.0f, 1.0f, 1.0f, 1.0f);

    state.mFrustumVertices.resize(108, 0.0f);
    state.mFrustumNormals.resize(108, 0.0f);

    Renderer::getRenderer()->createFrustum(state.mFrustumVertices, state.mFrustumNormals, &state.mFrustumVAO, &state.mFrustumVBO[0],
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

    Renderer::getRenderer()->createGrid(state.mGridVertices, &state.mGridVAO, &state.mGridVBO);
}

static void destroyGizmoRenderer(GizmoRendererState &state)
{
    Renderer::getRenderer()->destroyFrustum(&state.mFrustumVAO, &state.mFrustumVBO[0], &state.mFrustumVBO[1]);
    Renderer::getRenderer()->destroyGrid(&state.mGridVAO, &state.mGridVBO);
}

static void renderLineGizmos(World *world, Camera *camera, GizmoRendererState &state,
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

    Renderer::getRenderer()->createLine(vertices, colors, &lineVAO, &lineVBO[0], &lineVBO[1]);

    //Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
    camera->getNativeGraphicsMainFBO()->bind();
    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    glm::mat4 mvp = camera->getProjMatrix() * camera->getViewMatrix();

    //Renderer::getRenderer()->use(state.mLineShaderProgram);
    //Renderer::getRenderer()->setMat4(state.mLineShaderMVPLoc, mvp);
    state.mLineShaderProgram->bind();
    state.mLineShaderProgram->setMat4(state.mLineShaderMVPLoc, mvp);

    Renderer::getRenderer()->renderLines(lineVAO, 0, (int)vertices.size() / 3);

    //Renderer::getRenderer()->unbindFramebuffer();
    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->destroyLine(&lineVAO, &lineVBO[0], &lineVBO[1]);
}

static void renderSphereGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                       const std::vector<SphereGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    Renderer::getRenderer()->turnOn(Capability::Blending);
    Renderer::getRenderer()->setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    Transform *transform = camera->getComponent<Transform>();

    Mesh *mesh = world->getPrimtiveMesh(PrimitiveType::Sphere);

    //Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
    camera->getNativeGraphicsMainFBO()->bind();
    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Renderer::getRenderer()->bindVertexArray(mesh->getNativeGraphicsVAO());

    /*Renderer::getRenderer()->use(state.mGizmoShaderProgram);

    Renderer::getRenderer()->setVec3(state.mGizmoShaderLightPosLoc, transform->getPosition());
    Renderer::getRenderer()->setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    Renderer::getRenderer()->setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());*/
    state.mGizmoShaderProgram->bind();
    state.mGizmoShaderProgram->setVec3(state.mGizmoShaderLightPosLoc, transform->getPosition());
    state.mGizmoShaderProgram->setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    state.mGizmoShaderProgram->setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());

    for (size_t i = 0; i < gizmos.size(); i++)
    {
        glm::mat4 model = glm::translate(glm::mat4(1.0f), gizmos[i].mSphere.mCentre);
        model = glm::scale(model,
                           glm::vec3(gizmos[i].mSphere.mRadius, gizmos[i].mSphere.mRadius, gizmos[i].mSphere.mRadius));

        //Renderer::getRenderer()->setColor(state.mGizmoShaderColorLoc, gizmos[i].mColor);
        //Renderer::getRenderer()->setMat4(state.mGizmoShaderModelLoc, model);
        state.mGizmoShaderProgram->setColor(state.mGizmoShaderColorLoc, gizmos[i].mColor);
        state.mGizmoShaderProgram->setMat4(state.mGizmoShaderModelLoc, model);

        Renderer::getRenderer()->renderWithCurrentlyBoundVAO(0, (int)mesh->getVertices().size() / 3);
    }

    Renderer::getRenderer()->unbindVertexArray();
    //Renderer::getRenderer()->unbindFramebuffer();
    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->turnOff(Capability::Blending);
}

static void renderAABBGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                     const std::vector<AABBGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    Renderer::getRenderer()->turnOn(Capability::Blending);
    Renderer::getRenderer()->setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    Transform *transform = camera->getComponent<Transform>();

    Mesh *mesh = world->getPrimtiveMesh(PrimitiveType::Cube);

    //Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
    camera->getNativeGraphicsMainFBO()->bind();
    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);
 
    Renderer::getRenderer()->bindVertexArray(mesh->getNativeGraphicsVAO());

    /*Renderer::getRenderer()->use(state.mGizmoShaderProgram);

    Renderer::getRenderer()->setVec3(state.mGizmoShaderLightPosLoc, transform->getPosition());
    Renderer::getRenderer()->setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    Renderer::getRenderer()->setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());*/
    state.mGizmoShaderProgram->bind();
    state.mGizmoShaderProgram->setVec3(state.mGizmoShaderLightPosLoc, transform->getPosition());
    state.mGizmoShaderProgram->setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    state.mGizmoShaderProgram->setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());

    for (size_t i = 0; i < gizmos.size(); i++)
    {
        glm::mat4 model = glm::translate(glm::mat4(1.0f), gizmos[i].mAABB.mCentre);
        model = glm::scale(model, gizmos[i].mAABB.mSize);

        //Renderer::getRenderer()->setColor(state.mGizmoShaderColorLoc, gizmos[i].mColor);
        //Renderer::getRenderer()->setMat4(state.mGizmoShaderModelLoc, model);
        state.mGizmoShaderProgram->setColor(state.mGizmoShaderColorLoc, gizmos[i].mColor);
        state.mGizmoShaderProgram->setMat4(state.mGizmoShaderModelLoc, model);

        Renderer::getRenderer()->renderWithCurrentlyBoundVAO(0, (int)mesh->getVertices().size() / 3);
    }

    Renderer::getRenderer()->unbindVertexArray();   
    //Renderer::getRenderer()->unbindFramebuffer();
    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->turnOff(Capability::Blending);
}

static void renderPlaneGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                      const std::vector<PlaneGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    Renderer::getRenderer()->turnOn(Capability::Blending);
    Renderer::getRenderer()->setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    Transform *transform = camera->getComponent<Transform>();

    Mesh *mesh = world->getPrimtiveMesh(PrimitiveType::Plane);

    //Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
    camera->getNativeGraphicsMainFBO()->bind();
    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Renderer::getRenderer()->bindVertexArray(mesh->getNativeGraphicsVAO());

    /*Renderer::getRenderer()->use(state.mGizmoShaderProgram);

    Renderer::getRenderer()->setVec3(state.mGizmoShaderLightPosLoc, transform->getPosition());
    Renderer::getRenderer()->setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    Renderer::getRenderer()->setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());*/
    state.mGizmoShaderProgram->bind();
    state.mGizmoShaderProgram->setVec3(state.mGizmoShaderLightPosLoc, transform->getPosition());
    state.mGizmoShaderProgram->setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    state.mGizmoShaderProgram->setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());

    for (size_t i = 0; i < gizmos.size(); i++)
    {
        glm::mat4 model = glm::translate(glm::mat4(1.0f), gizmos[i].mPlane.mX0);
        glm::vec3 a = glm::vec3(0, 0, 1);
        glm::vec3 b = gizmos[i].mPlane.mNormal;
        float d = glm::dot(a, b);
        glm::vec3 c = glm::cross(a, b);
        float angle = glm::atan(glm::length(c), d);

        model = glm::rotate(model, angle, c);

        //Renderer::getRenderer()->setColor(state.mGizmoShaderColorLoc, gizmos[i].mColor);
        //Renderer::getRenderer()->setMat4(state.mGizmoShaderModelLoc, model);
        state.mGizmoShaderProgram->setColor(state.mGizmoShaderColorLoc, gizmos[i].mColor);
        state.mGizmoShaderProgram->setMat4(state.mGizmoShaderModelLoc, model);

        Renderer::getRenderer()->renderWithCurrentlyBoundVAO(0, (int)mesh->getVertices().size() / 3);
    }

    Renderer::getRenderer()->unbindVertexArray();    
    //Renderer::getRenderer()->unbindFramebuffer();
    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->turnOff(Capability::Blending);
}

static void renderShadedFrustumGizmo(World *world, Camera *camera, GizmoRendererState &state,
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

    /*Renderer::getRenderer()->use(state.mGizmoShaderProgram);
    Renderer::getRenderer()->setVec3(state.mGizmoShaderLightPosLoc, transform->getPosition());
    Renderer::getRenderer()->setMat4(state.mGizmoShaderModelLoc, glm::mat4(1.0f));
    Renderer::getRenderer()->setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    Renderer::getRenderer()->setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());
    Renderer::getRenderer()->setColor(state.mGizmoShaderColorLoc, gizmo.mColor);*/
    state.mGizmoShaderProgram->bind();
    state.mGizmoShaderProgram->setVec3(state.mGizmoShaderLightPosLoc, transform->getPosition());
    state.mGizmoShaderProgram->setMat4(state.mGizmoShaderModelLoc, glm::mat4(1.0f));
    state.mGizmoShaderProgram->setMat4(state.mGizmoShaderViewLoc, camera->getViewMatrix());
    state.mGizmoShaderProgram->setMat4(state.mGizmoShaderProjLoc, camera->getProjMatrix());
    state.mGizmoShaderProgram->setColor(state.mGizmoShaderColorLoc, gizmo.mColor);

    Renderer::getRenderer()->updateFrustum(state.mFrustumVertices, state.mFrustumNormals, state.mFrustumVBO[0],
                            state.mFrustumVBO[1]);

    Renderer::getRenderer()->renderWithCurrentlyBoundVAO(0, (int)state.mFrustumVertices.size() / 3);
}

static void renderWireframeFrustumGizmo(World *world, Camera *camera, GizmoRendererState &state,
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

    /*Renderer::getRenderer()->use(state.mLineShaderProgram);
    Renderer::getRenderer()->setMat4(state.mLineShaderMVPLoc, mvp);*/
    state.mLineShaderProgram->bind();
    state.mLineShaderProgram->setMat4(state.mLineShaderMVPLoc, mvp);

    Renderer::getRenderer()->updateFrustum(state.mFrustumVertices, state.mFrustumVBO[0]);
    
    Renderer::getRenderer()->renderLinesWithCurrentlyBoundVAO(0, 24);
}

static void renderFrustumGizmos(World *world, Camera *camera, GizmoRendererState &state,
                                        const std::vector<FrustumGizmo> &gizmos)
{
    if (gizmos.empty())
    {
        return;
    }

    Renderer::getRenderer()->turnOn(Capability::Blending);
    Renderer::getRenderer()->setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    //Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
    camera->getNativeGraphicsMainFBO()->bind();
    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    Renderer::getRenderer()->bindVertexArray(state.mFrustumVAO);

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

    Renderer::getRenderer()->unbindVertexArray();
    //Renderer::getRenderer()->unbindFramebuffer();
    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->turnOff(Capability::Blending);
}

static void renderGridGizmo(World *world, Camera *camera, GizmoRendererState &state)
{
    Renderer::getRenderer()->turnOn(Capability::Blending);
    Renderer::getRenderer()->turnOn(Capability::LineSmoothing);
    Renderer::getRenderer()->setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    //Renderer::getRenderer()->bindFramebuffer(camera->getNativeGraphicsMainFBO());
    camera->getNativeGraphicsMainFBO()->bind();
    Renderer::getRenderer()->setViewport(camera->getViewport().mX, camera->getViewport().mY, camera->getViewport().mWidth,
                          camera->getViewport().mHeight);

    glm::mat4 mvp = camera->getProjMatrix() * camera->getViewMatrix();

    /*Renderer::getRenderer()->use(state.mGridShaderProgram);
    Renderer::getRenderer()->setMat4(state.mGridShaderMVPLoc, mvp);
    Renderer::getRenderer()->setColor(state.mGridShaderColorLoc, state.mGridColor);*/
    state.mGridShaderProgram->bind();
    state.mGridShaderProgram->setMat4(state.mGridShaderMVPLoc, mvp);
    state.mGridShaderProgram->setColor(state.mGridShaderColorLoc, state.mGridColor);

    Renderer::getRenderer()->renderLines(0, (int)state.mGridVertices.size(), state.mGridVAO);

    //Renderer::getRenderer()->unbindFramebuffer();
    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->turnOff(Capability::Blending);
    Renderer::getRenderer()->turnOff(Capability::LineSmoothing);
}