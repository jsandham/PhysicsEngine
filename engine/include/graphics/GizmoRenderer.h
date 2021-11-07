#ifndef GIZMORENDERER_H__
#define GIZMORENDERER_H__

#include <vector>

#include "../core/AABB.h"
#include "../core/Color.h"
#include "../core/Frustum.h"
#include "../core/Input.h"
#include "../core/Line.h"
#include "../core/Plane.h"
#include "../core/Ray.h"
#include "../core/Sphere.h"

#include "Graphics.h"

namespace PhysicsEngine
{
class World;
class Camera;

struct LineGizmo
{
    Line mLine;
    Color mColor;

    LineGizmo()
    {
    }
    LineGizmo(const Line &line, const Color &color) : mLine(line), mColor(color)
    {
    }
};

struct SphereGizmo
{
    Sphere mSphere;
    Color mColor;

    SphereGizmo()
    {
    }
    SphereGizmo(const Sphere &sphere, const Color &color) : mSphere(sphere), mColor(color)
    {
    }
};

struct AABBGizmo
{
    AABB mAABB;
    Color mColor;
    bool mWireFrame;

    AABBGizmo() : mWireFrame(false)
    {
    }
    AABBGizmo(const AABB &aabb, const Color &color, bool wireframe) : mAABB(aabb), mColor(color), mWireFrame(wireframe)
    {
    }
};

struct FrustumGizmo
{
    Frustum mFrustum;
    Color mColor;
    bool mWireFrame;

    FrustumGizmo() : mWireFrame(false)
    {
    }
    FrustumGizmo(const Frustum &frustum, const Color &color, bool wireframe)
        : mFrustum(frustum), mColor(color), mWireFrame(wireframe)
    {
    }
};

struct PlaneGizmo
{
    Plane mPlane;
    Color mColor;
    glm::vec3 mExtents;
    bool mWireFrame;

    PlaneGizmo() : mWireFrame(false)
    {
    }
    PlaneGizmo(const Plane &plane, glm::vec3 extents, const Color &color, bool wireframe)
        : mPlane(plane), mColor(color), mExtents(extents), mWireFrame(wireframe)
    {
    }
};

class GizmoRenderer
{
  private:
    World *mWorld;

    GizmoRendererState mState;

    std::vector<LineGizmo> mLines;
    std::vector<AABBGizmo> mAABBs;
    std::vector<SphereGizmo> mSpheres;
    std::vector<FrustumGizmo> mFrustums;
    std::vector<PlaneGizmo> mPlanes;

  public:
    GizmoRenderer();
    ~GizmoRenderer();

    void init(World *world);
    void update(Camera *camera);
    void drawGrid(Camera* camera);

    void addToDrawList(const Line &line, const Color &color);
    void addToDrawList(const Ray &ray, float t, const Color &color);
    void addToDrawList(const Sphere& sphere, const Color& color);
    void addToDrawList(const AABB &aabb, const Color &color, bool wireframe = false);
    void addToDrawList(const Frustum &frustum, const Color &color, bool wireframe = false);
    void addToDrawList(const Plane &plane, const glm::vec3 &extents, const Color &color, bool wireframe = false);
    void clearDrawList();
};

void initializeGizmoRenderer(World *world, GizmoRendererState &state);
void destroyGizmoRenderer(GizmoRendererState &state);
void renderLineGizmos(World *world, Camera *camera, GizmoRendererState &state, const std::vector<LineGizmo> &gizmos);
void renderPlaneGizmos(World *world, Camera *camera, GizmoRendererState &state, const std::vector<PlaneGizmo> &gizmos);
void renderAABBGizmos(World *world, Camera *camera, GizmoRendererState &state, const std::vector<AABBGizmo> &gizmos);
void renderSphereGizmos(World *world, Camera *camera, GizmoRendererState &state,
                        const std::vector<SphereGizmo> &gizmos);
void renderFrustumGizmos(World *world, Camera *camera, GizmoRendererState &state,
                         const std::vector<FrustumGizmo> &gizmos);

void renderShadedFrustumGizmo(World *world, Camera *camera, GizmoRendererState &state, const FrustumGizmo &gizmo);
void renderWireframeFrustumGizmo(World *world, Camera *camera, GizmoRendererState &state, const FrustumGizmo &gizmo);

void renderGridGizmo(World *world, Camera *camera, GizmoRendererState &state);
} // namespace PhysicsEngine

#endif