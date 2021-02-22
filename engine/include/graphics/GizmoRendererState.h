#ifndef __GIZMO_RENDERER_STATE_H__
#define __GIZMO_RENDERER_STATE_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/Shader.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
struct GizmoRendererState
{
    Shader *mLineShader;
    int mLineShaderProgram;
    int mLineShaderMVPLoc;

    Shader *mGizmoShader;
    int mGizmoShaderProgram;
    int mGizmoShaderModelLoc;
    int mGizmoShaderViewLoc;
    int mGizmoShaderProjLoc;
    int mGizmoShaderColorLoc;
    int mGizmoShaderLightPosLoc;
};

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

    AABBGizmo()
    {
    }
    AABBGizmo(const AABB &aabb, const Color &color) : mAABB(aabb), mColor(color)
    {
    }
};

struct FrustumGizmo
{
    Frustum mFrustum;
    Color mColor;

    FrustumGizmo()
    {
    }
    FrustumGizmo(const Frustum &frustum, const Color &color) : mFrustum(frustum), mColor(color)
    {
    }
};

struct PlaneGizmo
{
    Plane mPlane;
    Color mColor;
    glm::vec3 mExtents;

    PlaneGizmo()
    {
    }
    PlaneGizmo(const Plane &plane, glm::vec3 extents, const Color &color)
        : mPlane(plane), mColor(color), mExtents(extents)
    {
    }
};
} // namespace PhysicsEngine

#endif