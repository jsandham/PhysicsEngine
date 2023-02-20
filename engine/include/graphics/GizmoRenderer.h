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

#include "Renderer.h"
#include "RendererShaders.h"

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

    LineShader *mLineShader;
    GizmoShader *mGizmoShader;
    GridShader *mGridShader;

    MeshHandle *mFrustumHandle;
    VertexBuffer *mFrustumVertexBuffer;
    VertexBuffer *mFrustumNormalBuffer;
    std::vector<float> mFrustumVertices;
    std::vector<float> mFrustumNormals;

    MeshHandle *mGridHandle;
    VertexBuffer *mGridVertexBuffer;
    std::vector<glm::vec3> mGridVertices;
    glm::vec3 mGridOrigin;
    Color mGridColor;

    std::vector<LineGizmo> mLines;
    std::vector<AABBGizmo> mAABBs;
    std::vector<SphereGizmo> mSpheres;
    std::vector<FrustumGizmo> mFrustums;
    std::vector<PlaneGizmo> mPlanes;

  public:
    GizmoRenderer();
    ~GizmoRenderer();
    GizmoRenderer(const GizmoRenderer& other) = delete;
    GizmoRenderer& operator=(const GizmoRenderer& other) = delete;

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

  private:
    void renderLineGizmos(Camera *camera);
    void renderPlaneGizmos(Camera *camera);
    void renderAABBGizmos(Camera *camera);
    void renderSphereGizmos(Camera *camera);
    void renderFrustumGizmos(Camera *camera);
    void renderShadedFrustumGizmo(Camera *camera, const FrustumGizmo &gizmo);
    void renderWireframeFrustumGizmo(Camera *camera, const FrustumGizmo &gizmo);
    void renderGridGizmo(Camera *camera);
};

} // namespace PhysicsEngine

#endif