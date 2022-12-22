#include "../../include/graphics/GizmoRenderer.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

GizmoRenderer::GizmoRenderer()
{
}

GizmoRenderer::~GizmoRenderer()
{
    Renderer::getRenderer()->destroyFrustum(&mState.mFrustumVAO, &mState.mFrustumVBO[0], &mState.mFrustumVBO[1]);
    Renderer::getRenderer()->destroyGrid(&mState.mGridVAO, &mState.mGridVBO);
}

void GizmoRenderer::init(World *world)
{
    mWorld = world;

    mLineShader = RendererShaders::getLineShader();
    mGizmoShader = RendererShaders::getGizmoShader();
    mGridShader = RendererShaders::getGridShader();

    mState.mGridColor = Color(1.0f, 1.0f, 1.0f, 1.0f);

    mState.mFrustumVertices.resize(108, 0.0f);
    mState.mFrustumNormals.resize(108, 0.0f);

    Renderer::getRenderer()->createFrustum(mState.mFrustumVertices, mState.mFrustumNormals, &mState.mFrustumVAO,
                                           &mState.mFrustumVBO[0], &mState.mFrustumVBO[1]);

    for (int i = -100; i < 100; i++)
    {
        glm::vec3 start = glm::vec3(i, 0.0f, -100.0f);
        glm::vec3 end = glm::vec3(i, 0.0f, 100.0f);

        mState.mGridVertices.push_back(start);
        mState.mGridVertices.push_back(end);
    }

    for (int i = -100; i < 100; i++)
    {
        glm::vec3 start = glm::vec3(-100.0f, 0.0f, i);
        glm::vec3 end = glm::vec3(100.0f, 0.0f, i);

        mState.mGridVertices.push_back(start);
        mState.mGridVertices.push_back(end);
    }

    Renderer::getRenderer()->createGrid(mState.mGridVertices, &mState.mGridVAO, &mState.mGridVBO);
}

void GizmoRenderer::update(Camera *camera)
{
    for (size_t i = 0; i < mWorld->mBoundingSpheres.size(); i++)
    {
        /*addToDrawList(mWorld->mBoundingSpheres[i], mWorld->mRenderObjects[i].culled ? Color::blue : Color::red);*/
        addToDrawList(mWorld->mBoundingSpheres[i], Color(0.0f, 0.0f, 1.0f, 0.3f) /*Color::blue*/);
    }

    renderLineGizmos(camera);
    renderPlaneGizmos(camera);
    renderAABBGizmos(camera);
    renderSphereGizmos(camera);
    renderFrustumGizmos(camera);
}

void GizmoRenderer::drawGrid(Camera* camera)
{
    renderGridGizmo(camera);
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

void GizmoRenderer::renderLineGizmos(Camera *camera)
{
    if (mLines.empty())
    {
        return;
    }

    std::vector<float> vertices(6 * mLines.size());

    for (size_t i = 0; i < mLines.size(); i++)
    {
        vertices[6 * i + 0] = mLines[i].mLine.mStart.x;
        vertices[6 * i + 1] = mLines[i].mLine.mStart.y;
        vertices[6 * i + 2] = mLines[i].mLine.mStart.z;

        vertices[6 * i + 3] = mLines[i].mLine.mEnd.x;
        vertices[6 * i + 4] = mLines[i].mLine.mEnd.y;
        vertices[6 * i + 5] = mLines[i].mLine.mEnd.z;
    }

    std::vector<float> colors(8 * mLines.size());

    for (size_t i = 0; i < mLines.size(); i++)
    {
        colors[8 * i + 0] = mLines[i].mColor.mR;
        colors[8 * i + 1] = mLines[i].mColor.mG;
        colors[8 * i + 2] = mLines[i].mColor.mB;
        colors[8 * i + 3] = mLines[i].mColor.mA;

        colors[8 * i + 4] = mLines[i].mColor.mR;
        colors[8 * i + 5] = mLines[i].mColor.mG;
        colors[8 * i + 6] = mLines[i].mColor.mB;
        colors[8 * i + 7] = mLines[i].mColor.mA;
    }

    unsigned int lineVAO;
    unsigned int lineVBO[2];

    Renderer::getRenderer()->createLine(vertices, colors, &lineVAO, &lineVBO[0], &lineVBO[1]);

    camera->getNativeGraphicsMainFBO()->bind();
    camera->getNativeGraphicsMainFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                    camera->getViewport().mWidth, camera->getViewport().mHeight);

    glm::mat4 mvp = camera->getProjMatrix() * camera->getViewMatrix();

    mLineShader->bind();
    mLineShader->setMVP(mvp);

    Renderer::getRenderer()->renderLines(lineVAO, 0, (int)vertices.size() / 3);

    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->destroyLine(&lineVAO, &lineVBO[0], &lineVBO[1]);
}

void GizmoRenderer::renderSphereGizmos(Camera *camera)
{
    if (mSpheres.empty())
    {
        return;
    }

    Renderer::getRenderer()->turnOn(Capability::Blending);
    Renderer::getRenderer()->setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    Transform *transform = camera->getComponent<Transform>();

    Mesh *mesh = mWorld->getPrimtiveMesh(PrimitiveType::Sphere);

    camera->getNativeGraphicsMainFBO()->bind();
    camera->getNativeGraphicsMainFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                    camera->getViewport().mWidth, camera->getViewport().mHeight);

    Renderer::getRenderer()->bindVertexArray(mesh->getNativeGraphicsVAO());

    mGizmoShader->bind();
    mGizmoShader->setLightPos(transform->getPosition());
    mGizmoShader->setView(camera->getViewMatrix());
    mGizmoShader->setProjection(camera->getProjMatrix());

    for (size_t i = 0; i < mSpheres.size(); i++)
    {
        glm::mat4 model = glm::translate(glm::mat4(1.0f), mSpheres[i].mSphere.mCentre);
        model = glm::scale(
            model, glm::vec3(mSpheres[i].mSphere.mRadius, mSpheres[i].mSphere.mRadius, mSpheres[i].mSphere.mRadius));

        mGizmoShader->setColor(mSpheres[i].mColor);
        mGizmoShader->setModel(model);

        Renderer::getRenderer()->renderWithCurrentlyBoundVAO(0, (int)mesh->getVertices().size() / 3);
    }

    Renderer::getRenderer()->unbindVertexArray();
 
    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->turnOff(Capability::Blending);
}

void GizmoRenderer::renderAABBGizmos(Camera *camera)
{
    if (mAABBs.empty())
    {
        return;
    }

    Renderer::getRenderer()->turnOn(Capability::Blending);
    Renderer::getRenderer()->setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    Transform *transform = camera->getComponent<Transform>();

    Mesh *mesh = mWorld->getPrimtiveMesh(PrimitiveType::Cube);

    camera->getNativeGraphicsMainFBO()->bind();
    camera->getNativeGraphicsMainFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                    camera->getViewport().mWidth, camera->getViewport().mHeight);
 
    Renderer::getRenderer()->bindVertexArray(mesh->getNativeGraphicsVAO());

    mGizmoShader->bind();
    mGizmoShader->setLightPos(transform->getPosition());
    mGizmoShader->setView(camera->getViewMatrix());
    mGizmoShader->setProjection(camera->getProjMatrix());

    for (size_t i = 0; i < mAABBs.size(); i++)
    {
        glm::mat4 model = glm::translate(glm::mat4(1.0f), mAABBs[i].mAABB.mCentre);
        model = glm::scale(model, mAABBs[i].mAABB.mSize);

        mGizmoShader->setColor(mAABBs[i].mColor);
        mGizmoShader->setModel(model);

        Renderer::getRenderer()->renderWithCurrentlyBoundVAO(0, (int)mesh->getVertices().size() / 3);
    }

    Renderer::getRenderer()->unbindVertexArray();   
    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->turnOff(Capability::Blending);
}

void GizmoRenderer::renderPlaneGizmos(Camera *camera)
{
    if (mPlanes.empty())
    {
        return;
    }

    Renderer::getRenderer()->turnOn(Capability::Blending);
    Renderer::getRenderer()->setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    Transform *transform = camera->getComponent<Transform>();

    Mesh *mesh = mWorld->getPrimtiveMesh(PrimitiveType::Plane);

    camera->getNativeGraphicsMainFBO()->bind();
    camera->getNativeGraphicsMainFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                    camera->getViewport().mWidth, camera->getViewport().mHeight);

    Renderer::getRenderer()->bindVertexArray(mesh->getNativeGraphicsVAO());

    mGizmoShader->bind();
    mGizmoShader->setLightPos(transform->getPosition());
    mGizmoShader->setView(camera->getViewMatrix());
    mGizmoShader->setProjection(camera->getProjMatrix());

    for (size_t i = 0; i < mPlanes.size(); i++)
    {
        glm::mat4 model = glm::translate(glm::mat4(1.0f), mPlanes[i].mPlane.mX0);
        glm::vec3 a = glm::vec3(0, 0, 1);
        glm::vec3 b = mPlanes[i].mPlane.mNormal;
        float d = glm::dot(a, b);
        glm::vec3 c = glm::cross(a, b);
        float angle = glm::atan(glm::length(c), d);

        model = glm::rotate(model, angle, c);

        mGizmoShader->setColor(mPlanes[i].mColor);
        mGizmoShader->setModel(model);

        Renderer::getRenderer()->renderWithCurrentlyBoundVAO(0, (int)mesh->getVertices().size() / 3);
    }

    Renderer::getRenderer()->unbindVertexArray();    
    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->turnOff(Capability::Blending);
}

void GizmoRenderer::renderShadedFrustumGizmo(Camera *camera, const FrustumGizmo &gizmo)
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
        mState.mFrustumVertices[3 * j + 0] = temp[j].x;
        mState.mFrustumVertices[3 * j + 1] = temp[j].y;
        mState.mFrustumVertices[3 * j + 2] = temp[j].z;
    }

    for (int j = 0; j < 6; j++)
    {
        glm::vec3 a = temp[6 * j + 1] - temp[6 * j];
        glm::vec3 b = temp[6 * j + 2] - temp[6 * j];

        glm::vec3 normal = glm::cross(a, b);

        for (int k = 0; k < 6; k++)
        {
            mState.mFrustumNormals[18 * j + 3 * k + 0] = normal.x;
            mState.mFrustumNormals[18 * j + 3 * k + 1] = normal.y;
            mState.mFrustumNormals[18 * j + 3 * k + 2] = normal.z;
        }
    }

    Transform *transform = camera->getComponent<Transform>();

    mGizmoShader->bind();
    mGizmoShader->setLightPos(transform->getPosition());
    mGizmoShader->setView(camera->getViewMatrix());
    mGizmoShader->setProjection(camera->getProjMatrix());
    mGizmoShader->setColor(gizmo.mColor);
    mGizmoShader->setModel(glm::mat4(1.0f));

    Renderer::getRenderer()->updateFrustum(mState.mFrustumVertices, mState.mFrustumNormals, mState.mFrustumVBO[0],
                            mState.mFrustumVBO[1]);

    Renderer::getRenderer()->renderWithCurrentlyBoundVAO(0, (int)mState.mFrustumVertices.size() / 3);
}

void GizmoRenderer::renderWireframeFrustumGizmo(Camera *camera, const FrustumGizmo &gizmo)
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
        mState.mFrustumVertices[3 * j + 0] = temp[j].x;
        mState.mFrustumVertices[3 * j + 1] = temp[j].y;
        mState.mFrustumVertices[3 * j + 2] = temp[j].z;
    }

    glm::mat4 mvp = camera->getProjMatrix() * camera->getViewMatrix();

    mLineShader->bind();
    mLineShader->setMVP(mvp);

    Renderer::getRenderer()->updateFrustum(mState.mFrustumVertices, mState.mFrustumVBO[0]);
    
    Renderer::getRenderer()->renderLinesWithCurrentlyBoundVAO(0, 24);
}

void GizmoRenderer::renderFrustumGizmos(Camera *camera)
{
    if (mFrustums.empty())
    {
        return;
    }

    Renderer::getRenderer()->turnOn(Capability::Blending);
    Renderer::getRenderer()->setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    camera->getNativeGraphicsMainFBO()->bind();
    camera->getNativeGraphicsMainFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                    camera->getViewport().mWidth, camera->getViewport().mHeight);

    Renderer::getRenderer()->bindVertexArray(mState.mFrustumVAO);

    for (size_t i = 0; i < mFrustums.size(); i++)
    {
        if (!mFrustums[i].mWireFrame)
        {
            renderShadedFrustumGizmo(camera, mFrustums[i]);
        }
        else
        {
            renderWireframeFrustumGizmo(camera, mFrustums[i]);
        }
    }

    Renderer::getRenderer()->unbindVertexArray();
    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->turnOff(Capability::Blending);
}

void GizmoRenderer::renderGridGizmo(Camera *camera)
{
    Renderer::getRenderer()->turnOn(Capability::Blending);
    Renderer::getRenderer()->turnOn(Capability::LineSmoothing);
    Renderer::getRenderer()->setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

    camera->getNativeGraphicsMainFBO()->bind();
    camera->getNativeGraphicsMainFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                    camera->getViewport().mWidth, camera->getViewport().mHeight);

    glm::mat4 mvp = camera->getProjMatrix() * camera->getViewMatrix();

    mGridShader->bind();
    mGridShader->setMVP(mvp);
    mGridShader->setColor(mState.mGridColor);

    Renderer::getRenderer()->renderLines(0, (int)mState.mGridVertices.size(), mState.mGridVAO);

    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->turnOff(Capability::Blending);
    Renderer::getRenderer()->turnOff(Capability::LineSmoothing);
}