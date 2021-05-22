#ifndef GIZMO_RENDERING_PASSES_H__
#define GIZMO_RENDERING_PASSES_H__

#include <GL/glew.h>
#include <gl/gl.h>
#include <vector>

#include "../components/Camera.h"

#include "../core/AABB.h"
#include "../core/Input.h"
#include "../core/Line.h"
#include "../core/Plane.h"
#include "../core/Ray.h"
#include "../core/Sphere.h"
#include "../core/World.h"

#include "../graphics/GizmoRendererState.h"

namespace PhysicsEngine
{
void initializeGizmoRenderer(World *world, GizmoRendererState &state);
void destroyGizmoRenderer(GizmoRendererState& state);
void renderLineGizmos(World *world, Camera *camera, GizmoRendererState &state, const std::vector<LineGizmo> &gizmos);
void renderPlaneGizmos(World *world, Camera *camera, GizmoRendererState &state, const std::vector<PlaneGizmo> &gizmos);
void renderAABBGizmos(World *world, Camera *camera, GizmoRendererState &state, const std::vector<AABBGizmo> &gizmos);
void renderSphereGizmos(World *world, Camera *camera, GizmoRendererState &state,
                        const std::vector<SphereGizmo> &gizmos);
void renderFrustumGizmos(World *world, Camera *camera, GizmoRendererState &state,
                         const std::vector<FrustumGizmo> &gizmos);

void renderShadedFrustumGizmo(World* world, Camera* camera, GizmoRendererState& state, const FrustumGizmo& gizmo);
void renderWireframeFrustumGizmo(World* world, Camera* camera, GizmoRendererState& state, const FrustumGizmo& gizmo);

void renderGridGizmo(World* world, Camera* camera, GizmoRendererState& state);
} // namespace PhysicsEngine

#endif