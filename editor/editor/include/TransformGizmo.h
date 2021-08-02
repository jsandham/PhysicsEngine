//#ifndef __TRANSFORM_GIZMOS_H__
//#define __TRANSFORM_GIZMOS_H__
//
//#define GLM_FORCE_RADIANS
//
//#include <GL/glew.h>
//#include <gl/gl.h>
//
////#include "glm/detail/func_trigonometric.hpp"
//#include "glm/glm.hpp"
//#include "glm/gtc/constants.hpp"
//
//#include "EditorCameraSystem.h"
//
//#include "systems/GizmoSystem.h"
//
//namespace PhysicsEditor
//{
//enum class Axis
//{
//    Axis_X = 0,
//    Axis_Y = 1,
//    Axis_Z = 2,
//    Axis_None = 3
//};
//
//enum class GizmoMode
//{
//    Translation = 0,
//    Rotation = 1,
//    Scale = 2
//};
//
//class TransformGizmo
//{
//  private:
//    GLuint mTranslationVAO[3];
//    GLuint mTranslationVBO[3];
//    GLuint mRotationVAO[3];
//    GLuint mRotationVBO[3];
//    GLuint mScaleVAO[3];
//    GLuint mScaleVBO[6];
//
//    GLuint mGizmoShaderProgram;
//    int mGizmoShaderMVPLoc;
//    int mGizmoShaderColorLoc;
//
//    GLuint mGizmo3dShaderProgram;
//    int mGizmo3dShaderModelLoc;
//    int mGizmo3dShaderViewLoc;
//    int mGizmo3dShaderProjLoc;
//    int mGizmo3dShaderColorLoc;
//    int mGizmo3dShaderLightPosLoc;
//
//    GizmoMode mode;
//    Axis highlightedTransformAxis;
//    Axis selectedTransformAxis;
//    glm::vec3 start;
//    glm::vec3 end;
//    glm::vec3 normal;
//
//  public:
//    void initialize();
//    void update(PhysicsEngine::EditorCameraSystem *cameraSystem, PhysicsEngine::GizmoSystem *gizmoSystem,
//                PhysicsEngine::Transform *selectedTransform, float mousePosX, float mousePosY, float contentWidth,
//                float contentHeight);
//    void setGizmoMode(GizmoMode mode);
//
//    bool isGizmoHighlighted() const;
//
//  private:
//    void updateTranslation(PhysicsEngine::EditorCameraSystem *cameraSystem, PhysicsEngine::Transform *selectedTransform,
//                           float mousePosX, float mousePosY, float contentWidth, float contentHeight);
//    void updateRotation(PhysicsEngine::EditorCameraSystem *cameraSystem, PhysicsEngine::GizmoSystem *gizmoSystem,
//                        PhysicsEngine::Transform *selectedTransform, float mousePosX, float mousePosY,
//                        float contentWidth, float contentHeight);
//    void updateScale(PhysicsEngine::EditorCameraSystem *cameraSystem, PhysicsEngine::Transform *selectedTransform,
//                     float mousePosX, float mousePosY, float contentWidth, float contentHeight);
//
//    void drawTranslation(const PhysicsEngine::Viewport &viewport, const glm::mat4 &projection, const glm::mat4 &view,
//                         const glm::mat4 &model, GLuint fbo, Axis highlightAxis, Axis selectedAxis);
//    void drawRotation(const PhysicsEngine::Viewport &viewport, const glm::mat4 &projection, const glm::mat4 &view,
//                      const glm::mat4 &model, GLuint fbo, Axis highlightAxis, Axis selectedAxis);
//    void drawScale(const PhysicsEngine::Viewport &viewport, const glm::mat4 &projection, const glm::mat4 &view,
//                   const glm::vec3 &cameraPos, const glm::mat4 &model, GLuint fbo, Axis highlightAxis,
//                   Axis selectedAxis);
//};
//} // namespace PhysicsEditor
//
//#endif