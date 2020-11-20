#include "../../include/graphics/GizmoRenderer.h"
#include "../../include/graphics/GizmoRendererPasses.h"

using namespace PhysicsEngine;

GizmoRenderer::GizmoRenderer()
{
}

GizmoRenderer::~GizmoRenderer()
{
}

void GizmoRenderer::init(World *world)
{
    mWorld = world;

    initializeGizmoRenderer(mWorld, mState);
}

void GizmoRenderer::update(Camera *camera)
{
    renderLineGizmos(mWorld, camera, mState, mLines);
    renderPlaneGizmos(mWorld, camera, mState, mPlanes);
    renderAABBGizmos(mWorld, camera, mState, mAABBs);
    renderSphereGizmos(mWorld, camera, mState, mSpheres);
    renderFrustumGizmos(mWorld, camera, mState, mFrustums);
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

void GizmoRenderer::addToDrawList(const AABB &aabb, const Color &color)
{
    mAABBs.push_back(AABBGizmo(aabb, color));
}

void GizmoRenderer::addToDrawList(const Sphere &sphere, const Color &color)
{
    mSpheres.push_back(SphereGizmo(sphere, color));
}

void GizmoRenderer::addToDrawList(const Frustum &frustum, const Color &color)
{
    mFrustums.push_back(FrustumGizmo(frustum, color));
}

void GizmoRenderer::addToDrawList(const Plane &plane, const glm::vec3 &extents, const Color &color)
{
    mPlanes.push_back(PlaneGizmo(plane, extents, color));
}

void GizmoRenderer::clearDrawList()
{
    mLines.clear();
    mAABBs.clear();
    mSpheres.clear();
    mFrustums.clear();
    mPlanes.clear();
}