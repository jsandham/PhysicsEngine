#include "../../include/graphics/GizmoRenderer.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

GizmoRenderer::GizmoRenderer()
{
    mFrustumVertexBuffer = VertexBuffer::create();
    mFrustumNormalBuffer = VertexBuffer::create();
    mFrustumHandle = MeshHandle::create();

    mGridVertexBuffer = VertexBuffer::create();
    mGridHandle = MeshHandle::create();
}

GizmoRenderer::~GizmoRenderer()
{
    delete mFrustumVertexBuffer;
    delete mFrustumNormalBuffer;
    delete mFrustumHandle;

    delete mGridVertexBuffer;
    delete mGridHandle;
}

void GizmoRenderer::init(World *world)
{
    mWorld = world;

    mLineShader = RendererShaders::getLineShader();
    mGizmoShader = RendererShaders::getGizmoShader();
    mGizmoInstancedShader = RendererShaders::getGizmoInstancedShader();
    mGridShader = RendererShaders::getGridShader();

    mGridColor = Color(1.0f, 1.0f, 1.0f, 1.0f);

    mFrustumVertices.resize(108, 0.0f);
    mFrustumNormals.resize(108, 0.0f);

    mFrustumVertexBuffer->bind();
    mFrustumVertexBuffer->resize(sizeof(float) * mFrustumVertices.size());
    mFrustumVertexBuffer->unbind();

    mFrustumNormalBuffer->bind();
    mFrustumNormalBuffer->resize(sizeof(float) * mFrustumNormals.size());
    mFrustumNormalBuffer->unbind();

    mFrustumHandle->addVertexBuffer(mFrustumVertexBuffer, AttribType::Vec3);
    mFrustumHandle->addVertexBuffer(mFrustumNormalBuffer, AttribType::Vec3);

    mGridVertices.reserve(800);

    for (int i = -100; i < 100; i++)
    {
        glm::vec3 start = glm::vec3(i, 0.0f, -100.0f);
        glm::vec3 end = glm::vec3(i, 0.0f, 100.0f);

        mGridVertices.push_back(start);
        mGridVertices.push_back(end);
    }

    for (int i = -100; i < 100; i++)
    {
        glm::vec3 start = glm::vec3(-100.0f, 0.0f, i);
        glm::vec3 end = glm::vec3(100.0f, 0.0f, i);

        mGridVertices.push_back(start);
        mGridVertices.push_back(end);
    }

    mGridVertexBuffer->bind();
    mGridVertexBuffer->resize(sizeof(glm::vec3) * mGridVertices.size());
    mGridVertexBuffer->setData(mGridVertices.data(), 0, sizeof(glm::vec3) * mGridVertices.size());
    mGridVertexBuffer->unbind();

    mGridHandle->addVertexBuffer(mGridVertexBuffer, AttribType::Vec3);
}

void GizmoRenderer::update(Camera *camera)
{
    for (size_t i = 0; i < mWorld->mBoundingSpheres.size(); i++)
    {
        addToDrawList(mWorld->mBoundingSpheres[i], mWorld->mFrustumVisible[i] ? Color(0.0f, 0.0f, 1.0f, 0.3f) : Color(1.0f, 0.0f, 0.0f, 0.3f));
    }

    renderLineGizmos(camera);
    renderPlaneGizmos(camera);
    renderAABBGizmos(camera);
    renderSphereGizmos(camera);
    renderFrustumGizmos(camera);
}

void GizmoRenderer::drawGrid(Camera *camera)
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

void GizmoRenderer::addToDrawList(const Sphere &sphere, const Color &color)
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

    VertexBuffer *lineVertexBuffer = VertexBuffer::create();
    VertexBuffer *lineColorBuffer = VertexBuffer::create();
    MeshHandle *lineHandle = MeshHandle::create();

    lineVertexBuffer->bind();
    lineVertexBuffer->resize(sizeof(float) * vertices.size());
    lineVertexBuffer->setData(vertices.data(), 0, sizeof(float) * vertices.size());
    lineVertexBuffer->unbind();

    lineColorBuffer->bind();
    lineColorBuffer->resize(sizeof(float) * colors.size());
    lineColorBuffer->setData(colors.data(), 0, sizeof(float) * colors.size());
    lineColorBuffer->unbind();

    lineHandle->addVertexBuffer(lineVertexBuffer, AttribType::Vec3);
    lineHandle->addVertexBuffer(lineColorBuffer, AttribType::Vec4);

    camera->getNativeGraphicsMainFBO()->bind();
    camera->getNativeGraphicsMainFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                    camera->getViewport().mWidth, camera->getViewport().mHeight);

    glm::mat4 mvp = camera->getProjMatrix() * camera->getViewMatrix();

    mLineShader->bind();
    mLineShader->setMVP(mvp);

    lineHandle->drawLines(0, vertices.size() / 3);

    camera->getNativeGraphicsMainFBO()->unbind();

    delete lineVertexBuffer;
    delete lineColorBuffer;
    delete lineHandle;
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

    if (mSpheres.size() < 100)
    {
        mGizmoShader->bind();
        mGizmoShader->setLightPos(transform->getPosition());
        mGizmoShader->setView(camera->getViewMatrix());
        mGizmoShader->setProjection(camera->getProjMatrix());

        for (size_t i = 0; i < mSpheres.size(); i++)
        {
            glm::mat4 model = glm::mat4(1.0f);
            model[3].x = mSpheres[i].mSphere.mCentre.x;
            model[3].y = mSpheres[i].mSphere.mCentre.y;
            model[3].z = mSpheres[i].mSphere.mCentre.z;

            model[0].x = mSpheres[i].mSphere.mRadius;
            model[1].y = mSpheres[i].mSphere.mRadius;
            model[2].z = mSpheres[i].mSphere.mRadius;

            mGizmoShader->setColor(mSpheres[i].mColor);
            mGizmoShader->setModel(model);

            mesh->getNativeGraphicsHandle()->drawIndexed(0, mesh->getIndices().size());
        }
    }
    else
    {
        mGizmoInstancedShader->bind();
        mGizmoInstancedShader->setLightPos(transform->getPosition());
        mGizmoInstancedShader->setView(camera->getViewMatrix());
        mGizmoInstancedShader->setProjection(camera->getProjMatrix());

        std::vector<glm::mat4> models(mSpheres.size());
        for (size_t i = 0; i < models.size(); i++)
        {
            models[i] = glm::mat4(1.0f);
            models[i][3].x = mSpheres[i].mSphere.mCentre.x;
            models[i][3].y = mSpheres[i].mSphere.mCentre.y;
            models[i][3].z = mSpheres[i].mSphere.mCentre.z;

            models[i][0].x = mSpheres[i].mSphere.mRadius;
            models[i][1].y = mSpheres[i].mSphere.mRadius;
            models[i][2].z = mSpheres[i].mSphere.mRadius;
        }

        std::vector<glm::uvec4> colors(mSpheres.size());
        for (size_t i = 0; i < colors.size(); i++)
        {
            Color32 c = Color32::convertColorToColor32(mSpheres[i].mColor);

            colors[i].r = c.mR;
            colors[i].g = c.mG;
            colors[i].b = c.mB;
            colors[i].a = c.mA;
        }

        VertexBuffer *modelBuffer = mesh->getNativeGraphicsInstanceModelBuffer();
        VertexBuffer *colorBuffer = mesh->getNativeGraphicsInstanceColorBuffer();

        modelBuffer->bind();
        modelBuffer->resize(sizeof(glm::mat4) * models.size());
        modelBuffer->setData(models.data(), 0, sizeof(glm::mat4) * models.size());
        modelBuffer->unbind();

        colorBuffer->bind();
        colorBuffer->resize(sizeof(glm::uvec4) * colors.size());
        colorBuffer->setData(colors.data(), 0, sizeof(glm::uvec4) * colors.size());
        colorBuffer->unbind();

        Renderer::getRenderer()->drawIndexedInstanced(mesh->getNativeGraphicsHandle(), mesh->getSubMeshStartIndex(0),
                                                      (mesh->getSubMeshEndIndex(0) - mesh->getSubMeshStartIndex(0)),
                                                      mSpheres.size(), camera->mQuery);

    }

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

        mesh->getNativeGraphicsHandle()->drawIndexed(0, mesh->getIndices().size());
    }

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

        mesh->getNativeGraphicsHandle()->drawIndexed(0, mesh->getIndices().size());
    }

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
        mFrustumVertices[3 * j + 0] = temp[j].x;
        mFrustumVertices[3 * j + 1] = temp[j].y;
        mFrustumVertices[3 * j + 2] = temp[j].z;
    }

    for (int j = 0; j < 6; j++)
    {
        glm::vec3 a = temp[6 * j + 1] - temp[6 * j];
        glm::vec3 b = temp[6 * j + 2] - temp[6 * j];

        glm::vec3 normal = glm::cross(a, b);

        for (int k = 0; k < 6; k++)
        {
            mFrustumNormals[18 * j + 3 * k + 0] = normal.x;
            mFrustumNormals[18 * j + 3 * k + 1] = normal.y;
            mFrustumNormals[18 * j + 3 * k + 2] = normal.z;
        }
    }

    Transform *transform = camera->getComponent<Transform>();

    mGizmoShader->bind();
    mGizmoShader->setLightPos(transform->getPosition());
    mGizmoShader->setView(camera->getViewMatrix());
    mGizmoShader->setProjection(camera->getProjMatrix());
    mGizmoShader->setColor(gizmo.mColor);
    mGizmoShader->setModel(glm::mat4(1.0f));

    mFrustumVertexBuffer->bind();
    mFrustumVertexBuffer->setData(mFrustumVertices.data(), 0, sizeof(float) * mFrustumVertices.size());
    mFrustumVertexBuffer->unbind();

    mFrustumNormalBuffer->bind();
    mFrustumNormalBuffer->setData(mFrustumNormals.data(), 0, sizeof(float) * mFrustumNormals.size());
    mFrustumNormalBuffer->unbind();

    mFrustumHandle->draw(0, mFrustumVertices.size() / 3);
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
        mFrustumVertices[3 * j + 0] = temp[j].x;
        mFrustumVertices[3 * j + 1] = temp[j].y;
        mFrustumVertices[3 * j + 2] = temp[j].z;
    }

    glm::mat4 mvp = camera->getProjMatrix() * camera->getViewMatrix();

    mLineShader->bind();
    mLineShader->setMVP(mvp);

    mFrustumVertexBuffer->bind();
    mFrustumVertexBuffer->setData(mFrustumVertices.data(), 0, sizeof(float) * mFrustumVertices.size());
    mFrustumVertexBuffer->unbind();

    mFrustumHandle->drawLines(0, 24);
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

    mFrustumHandle->bind();

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

    mFrustumHandle->unbind();
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
    mGridShader->setColor(mGridColor);

    mGridHandle->drawLines(0, mGridVertices.size());

    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->turnOff(Capability::Blending);
    Renderer::getRenderer()->turnOff(Capability::LineSmoothing);
}