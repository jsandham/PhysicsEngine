#ifndef __GIZMORENDERER_H__
#define __GIZMORENDERER_H__

#include <GL/glew.h>
#include <gl/gl.h>
#include <vector>

#include "../components/Camera.h"

#include "../core/AABB.h"
#include "../core/Color.h"
#include "../core/Frustum.h"
#include "../core/Input.h"
#include "../core/Line.h"
#include "../core/Plane.h"
#include "../core/Ray.h"
#include "../core/Sphere.h"

#include "../graphics/GizmoRendererState.h"

namespace PhysicsEngine
{
class World;

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

    void addToDrawList(const Line &line, const Color &color);
    void addToDrawList(const Ray &ray, float t, const Color &color);
    void addToDrawList(const AABB &aabb, const Color &color);
    void addToDrawList(const Sphere &sphere, const Color &color);
    void addToDrawList(const Frustum &frustum, const Color &color);
    void addToDrawList(const Plane &plane, const glm::vec3 &extents, const Color &color);
    void clearDrawList();
};
} // namespace PhysicsEngine

#endif