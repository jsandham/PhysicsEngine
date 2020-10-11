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
        Shader* mGizmoShader;
        int mGizmoShaderProgram;
        int mGizmoShaderModelLoc;
        int mGizmoShaderViewLoc;
        int mGizmoShaderProjLoc;
        int mGizmoShaderColorLoc;
        int mGizmoShaderLightPosLoc;

        int mGizmoShaderMVPLoc;

        // lines & rays 
        GLuint mLineVAO;
        GLuint mLineVBO;

        // sphere
        GLuint mSphereVAO;
        GLuint mSphereVBO;

        // AABB
        GLuint mAABBVAO;
        GLuint mAABBVBO;
    };

    struct LineGizmo
    {
        Line mLine;
        Color mColor;

        LineGizmo() {}
        LineGizmo(const Line& line, const Color& color) : mLine(line), mColor(color) {}
    };

    struct SphereGizmo
    {
        Sphere mSphere;
        Color mColor;

        SphereGizmo() {}
        SphereGizmo(const Sphere& sphere, const Color& color) : mSphere(sphere), mColor(color) {}
    };

    struct AABBGizmo
    {
        AABB mAABB;
        Color mColor;

        AABBGizmo() {}
        AABBGizmo(const AABB& aabb, const Color& color) : mAABB(aabb), mColor(color) {}
    };

    struct FrustumGizmo
    {
        Frustum mFrustum;
        Color mColor;
        glm::vec3 mPosition;
        glm::vec3 mFront;
        glm::vec3 mUp;
        glm::vec3 mRight;

        FrustumGizmo() {}
        FrustumGizmo(const Frustum& frustum, const glm::vec3 &pos, const glm::vec3 &front, 
            const glm::vec3 &up, const glm::vec3 &right, const Color& color) 
            : mFrustum(frustum), mFront(front), mUp(up), mRight(right), mColor(color) {}
    };
} // namespace PhysicsEngine

#endif