#ifndef __MESH_DRAWER_H__
#define __MESH_DRAWER_H__

#include "../EditorClipboard.h"
#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class MeshDrawer : public InspectorDrawer
{
  private:
    GLuint mFBO;
    GLuint mColor;
    GLuint mDepth;

    CameraUniform cameraUniform;

    glm::mat4 model;

    float mouseX;
    float mouseY;

    int activeDrawModeIndex;
    bool wireframeOn;
    bool resetModelMatrix;

  public:
    MeshDrawer();
    ~MeshDrawer();

    void render(EditorClipboard &clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif