#include "../../include/graphics/GizmoRenderer.h"
#include "../../include/core/World.h"
#include "../../include/core/Intersect.h"

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

    mFrustumVertexBuffer->bind(0);
    mFrustumVertexBuffer->resize(sizeof(float) * mFrustumVertices.size());
    mFrustumVertexBuffer->unbind(0);

    mFrustumNormalBuffer->bind(1);
    mFrustumNormalBuffer->resize(sizeof(float) * mFrustumNormals.size());
    mFrustumNormalBuffer->unbind(1);

    mFrustumHandle->addVertexBuffer(mFrustumVertexBuffer, "POSITION", AttribType::Vec3);
    mFrustumHandle->addVertexBuffer(mFrustumNormalBuffer, "NORMAL", AttribType::Vec3);

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

    mGridVertexBuffer->bind(0);
    mGridVertexBuffer->resize(sizeof(glm::vec3) * mGridVertices.size());
    mGridVertexBuffer->setData(mGridVertices.data(), 0, sizeof(glm::vec3) * mGridVertices.size());
    mGridVertexBuffer->unbind(0);

    mGridHandle->addVertexBuffer(mGridVertexBuffer, "POSITION", AttribType::Vec3);
}

void GizmoRenderer::update(Camera *camera)
{
    //addToDrawList(mWorld->mBoundingVolume, Color(1.0f, 0.91764705f, 0.01568627f, 0.3f), true);

    //renderLineGizmos(camera);
    //renderPlaneGizmos(camera);
    //renderAABBGizmos(camera);
    //renderSphereGizmos(camera);
    //renderFrustumGizmos(camera);

    //renderBoundingSpheres(camera);
    renderBoundingAABBs(camera);
    //renderBoundingVolumeHeirarchy(camera);
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

    lineVertexBuffer->bind(0);
    lineVertexBuffer->resize(sizeof(float) * vertices.size());
    lineVertexBuffer->setData(vertices.data(), 0, sizeof(float) * vertices.size());
    lineVertexBuffer->unbind(0);

    lineColorBuffer->bind(1);
    lineColorBuffer->resize(sizeof(float) * colors.size());
    lineColorBuffer->setData(colors.data(), 0, sizeof(float) * colors.size());
    lineColorBuffer->unbind(1);

    lineHandle->addVertexBuffer(lineVertexBuffer, "POSITION", AttribType::Vec3);
    lineHandle->addVertexBuffer(lineColorBuffer, "COLOR", AttribType::Vec4);

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

        modelBuffer->bind(3);
        modelBuffer->resize(sizeof(glm::mat4) * models.size());
        modelBuffer->setData(models.data(), 0, sizeof(glm::mat4) * models.size());
        modelBuffer->unbind(3);

        colorBuffer->bind(7);
        colorBuffer->resize(sizeof(glm::uvec4) * colors.size());
        colorBuffer->setData(colors.data(), 0, sizeof(glm::uvec4) * colors.size());
        colorBuffer->unbind(7);

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

    mFrustumVertexBuffer->bind(0);
    mFrustumVertexBuffer->setData(mFrustumVertices.data(), 0, sizeof(float) * mFrustumVertices.size());
    mFrustumVertexBuffer->unbind(0);

    mFrustumNormalBuffer->bind(1);
    mFrustumNormalBuffer->setData(mFrustumNormals.data(), 0, sizeof(float) * mFrustumNormals.size());
    mFrustumNormalBuffer->unbind(1);

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

    mFrustumVertexBuffer->bind(0);
    mFrustumVertexBuffer->setData(mFrustumVertices.data(), 0, sizeof(float) * mFrustumVertices.size());
    mFrustumVertexBuffer->unbind(0);

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

void GizmoRenderer::renderBoundingSpheres(Camera *camera)
{
    if (mWorld->mBoundingSpheres.empty())
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

    mGizmoInstancedShader->bind();
    mGizmoInstancedShader->setLightPos(transform->getPosition());
    mGizmoInstancedShader->setView(camera->getViewMatrix());
    mGizmoInstancedShader->setProjection(camera->getProjMatrix());

    std::vector<Sphere> &spheres = mWorld->mBoundingSpheres;
    std::vector<int> &visible = mWorld->mFrustumVisible;

    std::vector<glm::mat4> models(spheres.size(), glm::mat4(0.0f));
    
    for (size_t i = 0; i < models.size(); i++)
    {
        models[i][3].x = spheres[i].mCentre.x;
        models[i][3].y = spheres[i].mCentre.y;
        models[i][3].z = spheres[i].mCentre.z;

        models[i][0].x = spheres[i].mRadius;
        models[i][1].y = spheres[i].mRadius;
        models[i][2].z = spheres[i].mRadius;
        models[i][3].w = 1.0f;
    }

    static glm::uvec4 blue = glm::uvec4(0, 0, 255, 78);
    static glm::uvec4 red = glm::uvec4(255, 0, 0, 78);

    std::vector<glm::uvec4> colors(spheres.size());
    for (size_t i = 0; i < colors.size(); i++)
    {
        colors[i] = visible[i] ? blue : red;
    }

    VertexBuffer *modelBuffer = mesh->getNativeGraphicsInstanceModelBuffer();
    VertexBuffer *colorBuffer = mesh->getNativeGraphicsInstanceColorBuffer();

    modelBuffer->bind(3);
    modelBuffer->resize(sizeof(glm::mat4) * models.size());
    modelBuffer->setData(models.data(), 0, sizeof(glm::mat4) * models.size());
    modelBuffer->unbind(3);

    colorBuffer->bind(7);
    colorBuffer->resize(sizeof(glm::uvec4) * colors.size());
    colorBuffer->setData(colors.data(), 0, sizeof(glm::uvec4) * colors.size());
    colorBuffer->unbind(7);

    Renderer::getRenderer()->drawIndexedInstanced(mesh->getNativeGraphicsHandle(), mesh->getSubMeshStartIndex(0),
                                                    (mesh->getSubMeshEndIndex(0) - mesh->getSubMeshStartIndex(0)),
                                                    spheres.size(), camera->mQuery);

    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->turnOff(Capability::Blending);
}


void GizmoRenderer::renderBoundingAABBs(Camera *camera)
{
    if (mWorld->mBoundingAABBs.empty())
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

    mGizmoInstancedShader->bind();
    mGizmoInstancedShader->setLightPos(transform->getPosition());
    mGizmoInstancedShader->setView(camera->getViewMatrix());
    mGizmoInstancedShader->setProjection(camera->getProjMatrix());

    std::vector<AABB> &aabbs = mWorld->mBoundingAABBs;
    std::vector<int> &visible = mWorld->mFrustumVisible;

    std::vector<glm::mat4> models(aabbs.size(), glm::mat4(0.0f));

    for (size_t i = 0; i < models.size(); i++)
    {
        models[i][3].x = aabbs[i].mCentre.x;
        models[i][3].y = aabbs[i].mCentre.y;
        models[i][3].z = aabbs[i].mCentre.z;

        models[i][0].x = aabbs[i].mSize.x;
        models[i][1].y = aabbs[i].mSize.y;
        models[i][2].z = aabbs[i].mSize.z;
        models[i][3].w = 1.0f;
    }

    static glm::uvec4 blue = glm::uvec4(0, 0, 255, 78);
    static glm::uvec4 red = glm::uvec4(255, 0, 0, 78);

    std::vector<glm::uvec4> colors(aabbs.size());
    for (size_t i = 0; i < colors.size(); i++)
    {
        colors[i] = visible[i] ? blue : red;
    }

    VertexBuffer *modelBuffer = mesh->getNativeGraphicsInstanceModelBuffer();
    VertexBuffer *colorBuffer = mesh->getNativeGraphicsInstanceColorBuffer();

    modelBuffer->bind(3);
    modelBuffer->resize(sizeof(glm::mat4) * models.size());
    modelBuffer->setData(models.data(), 0, sizeof(glm::mat4) * models.size());
    modelBuffer->unbind(3);

    colorBuffer->bind(7);
    colorBuffer->resize(sizeof(glm::uvec4) * colors.size());
    colorBuffer->setData(colors.data(), 0, sizeof(glm::uvec4) * colors.size());
    colorBuffer->unbind(7);

    Renderer::getRenderer()->drawIndexedInstanced(mesh->getNativeGraphicsHandle(), mesh->getSubMeshStartIndex(0),
                                                  (mesh->getSubMeshEndIndex(0) - mesh->getSubMeshStartIndex(0)),
                                                  aabbs.size(), camera->mQuery);

    camera->getNativeGraphicsMainFBO()->unbind();

    Renderer::getRenderer()->turnOff(Capability::Blending);
}



void GizmoRenderer::renderBoundingVolumeHeirarchy(Camera *camera)
{
    RenderSystem *renderSystem = mWorld->getSystem<RenderSystem>();
    if (renderSystem != nullptr)
    {
        Renderer::getRenderer()->turnOn(Capability::Blending);
        Renderer::getRenderer()->setBlending(BlendingFactor::SRC_ALPHA, BlendingFactor::ONE_MINUS_SRC_ALPHA);

        const BVH &bvh = renderSystem->getBVH();

        std::vector<glm::vec3> mLineVertices(36 * bvh.mNodes.size());

        for (size_t i = 0; i < bvh.mNodes.size(); i++)
        {
            glm::vec3 min = bvh.mNodes[i].mMin;
            glm::vec3 max = bvh.mNodes[i].mMax;

            glm::vec3 size = max - min;

            glm::vec3 xsize = glm::vec3(size.x, 0.0f, 0.0f);
            glm::vec3 ysize = glm::vec3(0.0f, size.y, 0.0f);
            glm::vec3 zsize = glm::vec3(0.0f, 0.0f, size.z);

            //      v7*--------*v6
            //      . |       .|
            //   .    |     .  |
            //v3*----------*v2 |
            //  |     |    |   |
            //  |   v4|----|---|v5
            //  |   .      |  .
            //  |  .       | .
            //  | .        |.
            //  *----------*
            //  v0         v1
            // 
            // v0 = min, v6 = max
            glm::vec3 v0 = min;
            glm::vec3 v1 = min + xsize;
            glm::vec3 v2 = min + xsize + ysize;
            glm::vec3 v3 = min + ysize;

            glm::vec3 v4 = min + zsize;
            glm::vec3 v5 = min + zsize + xsize;
            glm::vec3 v6 = min + zsize + xsize + ysize;
            glm::vec3 v7 = min + zsize + ysize;

            // Using clockwise winding order
            /*mLineVertices[36 * i + 0] = v0;
            mLineVertices[36 * i + 1] = v2;
            mLineVertices[36 * i + 2] = v1;
            mLineVertices[36 * i + 3] = v0;
            mLineVertices[36 * i + 4] = v3;
            mLineVertices[36 * i + 5] = v2;

            mLineVertices[36 * i + 6] = v1;
            mLineVertices[36 * i + 7] = v6;
            mLineVertices[36 * i + 8] = v5;
            mLineVertices[36 * i + 9] = v1;
            mLineVertices[36 * i + 10] = v2;
            mLineVertices[36 * i + 11] = v6;

            mLineVertices[36 * i + 12] = v5;
            mLineVertices[36 * i + 13] = v7;
            mLineVertices[36 * i + 14] = v4;
            mLineVertices[36 * i + 15] = v5;
            mLineVertices[36 * i + 16] = v6;
            mLineVertices[36 * i + 17] = v7;

            mLineVertices[36 * i + 18] = v4;
            mLineVertices[36 * i + 19] = v3;
            mLineVertices[36 * i + 20] = v0;
            mLineVertices[36 * i + 21] = v4;
            mLineVertices[36 * i + 22] = v7;
            mLineVertices[36 * i + 23] = v3;

            mLineVertices[36 * i + 24] = v3;
            mLineVertices[36 * i + 25] = v6;
            mLineVertices[36 * i + 26] = v2;
            mLineVertices[36 * i + 27] = v3;
            mLineVertices[36 * i + 28] = v7;
            mLineVertices[36 * i + 29] = v6;

            mLineVertices[36 * i + 30] = v0;
            mLineVertices[36 * i + 31] = v1;
            mLineVertices[36 * i + 32] = v5;
            mLineVertices[36 * i + 33] = v0;
            mLineVertices[36 * i + 34] = v5;
            mLineVertices[36 * i + 35] = v4;*/


            // Using counter-clockwise winding order
            mLineVertices[36 * i + 0] = v0;
            mLineVertices[36 * i + 1] = v1;
            mLineVertices[36 * i + 2] = v2;
            mLineVertices[36 * i + 3] = v0;
            mLineVertices[36 * i + 4] = v2;
            mLineVertices[36 * i + 5] = v3;

            mLineVertices[36 * i + 6] = v1;
            mLineVertices[36 * i + 7] = v5;
            mLineVertices[36 * i + 8] = v6;
            mLineVertices[36 * i + 9] = v1;
            mLineVertices[36 * i + 10] = v6;
            mLineVertices[36 * i + 11] = v2;

            mLineVertices[36 * i + 12] = v5;
            mLineVertices[36 * i + 13] = v4;
            mLineVertices[36 * i + 14] = v7;
            mLineVertices[36 * i + 15] = v5;
            mLineVertices[36 * i + 16] = v7;
            mLineVertices[36 * i + 17] = v6;

            mLineVertices[36 * i + 18] = v4;
            mLineVertices[36 * i + 19] = v0;
            mLineVertices[36 * i + 20] = v3;
            mLineVertices[36 * i + 21] = v4;
            mLineVertices[36 * i + 22] = v3;
            mLineVertices[36 * i + 23] = v7;

            mLineVertices[36 * i + 24] = v3;
            mLineVertices[36 * i + 25] = v2;
            mLineVertices[36 * i + 26] = v6;
            mLineVertices[36 * i + 27] = v3;
            mLineVertices[36 * i + 28] = v6;
            mLineVertices[36 * i + 29] = v7;

            mLineVertices[36 * i + 30] = v0;
            mLineVertices[36 * i + 31] = v5;
            mLineVertices[36 * i + 32] = v1;
            mLineVertices[36 * i + 33] = v0;
            mLineVertices[36 * i + 34] = v4;
            mLineVertices[36 * i + 35] = v5;


            // Using counter-clockwise winding order
            /*for (int j = 0; j < 3; j++)
            {
                mLineVertices[3 * 36 * i + 3 * 0 + j] = v0[j];
                mLineVertices[3 * 36 * i + 3 * 1 + j] = v1[j];
                mLineVertices[3 * 36 * i + 3 * 2 + j] = v2[j];
                mLineVertices[3 * 36 * i + 3 * 3 + j] = v0[j];
                mLineVertices[3 * 36 * i + 3 * 4 + j] = v2[j];
                mLineVertices[3 * 36 * i + 3 * 5 + j] = v3[j];

                mLineVertices[3 * 36 * i + 3 * 6 + j] = v1[j];
                mLineVertices[3 * 36 * i + 3 * 7 + j] = v5[j];
                mLineVertices[3 * 36 * i + 3 * 8 + j] = v6[j];
                mLineVertices[3 * 36 * i + 3 * 9 + j] = v1[j];
                mLineVertices[3 * 36 * i + 3 * 10 + j] = v6[j];
                mLineVertices[3 * 36 * i + 3 * 11 + j] = v2[j];

                mLineVertices[3 * 36 * i + 3 * 12 + j] = v5[j];
                mLineVertices[3 * 36 * i + 3 * 13 + j] = v4[j];
                mLineVertices[3 * 36 * i + 3 * 14 + j] = v7[j];
                mLineVertices[3 * 36 * i + 3 * 15 + j] = v5[j];
                mLineVertices[3 * 36 * i + 3 * 16 + j] = v7[j];
                mLineVertices[3 * 36 * i + 3 * 17 + j] = v6[j];

                mLineVertices[3 * 36 * i + 3 * 18 + j] = v4[j];
                mLineVertices[3 * 36 * i + 3 * 19 + j] = v0[j];
                mLineVertices[3 * 36 * i + 3 * 20 + j] = v3[j];
                mLineVertices[3 * 36 * i + 3 * 21 + j] = v4[j];
                mLineVertices[3 * 36 * i + 3 * 22 + j] = v3[j];
                mLineVertices[3 * 36 * i + 3 * 23 + j] = v7[j];

                mLineVertices[3 * 36 * i + 3 * 24 + j] = v3[j];
                mLineVertices[3 * 36 * i + 3 * 25 + j] = v2[j];
                mLineVertices[3 * 36 * i + 3 * 26 + j] = v6[j];
                mLineVertices[3 * 36 * i + 3 * 27 + j] = v3[j];
                mLineVertices[3 * 36 * i + 3 * 28 + j] = v6[j];
                mLineVertices[3 * 36 * i + 3 * 29 + j] = v7[j];

                mLineVertices[3 * 36 * i + 3 * 30 + j] = v0[j];
                mLineVertices[3 * 36 * i + 3 * 31 + j] = v5[j];
                mLineVertices[3 * 36 * i + 3 * 32 + j] = v1[j];
                mLineVertices[3 * 36 * i + 3 * 33 + j] = v0[j];
                mLineVertices[3 * 36 * i + 3 * 34 + j] = v4[j];
                mLineVertices[3 * 36 * i + 3 * 35 + j] = v5[j];
            }*/
        }

        MeshHandle *meshHandle = MeshHandle::create();
        VertexBuffer *vertexBuffer = VertexBuffer::create();

        vertexBuffer->bind(0);
        vertexBuffer->resize(mLineVertices.size() * sizeof(glm::vec3));
        vertexBuffer->setData(mLineVertices.data(), 0, mLineVertices.size() * sizeof(glm::vec3));
        vertexBuffer->unbind(0);

        meshHandle->addVertexBuffer(vertexBuffer, "POSITION", AttribType::Vec3);

        camera->getNativeGraphicsMainFBO()->bind();
        camera->getNativeGraphicsMainFBO()->setViewport(camera->getViewport().mX, camera->getViewport().mY,
                                                        camera->getViewport().mWidth, camera->getViewport().mHeight);

        Transform *transform = camera->getComponent<Transform>();

        glm::mat4 mvp = camera->getProjMatrix() * camera->getViewMatrix();

        mGridShader->bind();
        mGridShader->setColor(Color(1.0f, 0.91764705f, 0.01568627f, 0.5f));
        mGridShader->setMVP(mvp);

        Renderer::getRenderer()->draw(meshHandle, 0, mLineVertices.size(), camera->mQuery);

        camera->getNativeGraphicsMainFBO()->unbind();
    
        Renderer::getRenderer()->turnOff(Capability::Blending);

        delete vertexBuffer;
        delete meshHandle;
    }
}