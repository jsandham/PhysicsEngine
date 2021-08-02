#ifndef GIZMO_RENDERER_STATE_H__
#define GIZMO_RENDERER_STATE_H__

#include <GL/glew.h>
#include <gl/gl.h>
#include <vector>

#include "../core/Shader.h"

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

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

    Shader* mGridShader;
    int mGridShaderProgram;
    int mGridShaderMVPLoc;
    int mGridShaderColorLoc;



    GLuint mFrustumVAO;
    GLuint mFrustumVBO[2];
    std::vector<float> mFrustumVertices;
    std::vector<float> mFrustumNormals;

    GLuint mGridVAO;
    GLuint mGridVBO;
    std::vector<glm::vec3> mGridVertices;
    glm::vec3 mGridOrigin;
    Color mGridColor;
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
    FrustumGizmo(const Frustum &frustum, const Color &color, bool wireframe) : mFrustum(frustum), mColor(color), mWireFrame(wireframe)
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
} // namespace PhysicsEngine

#endif