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

    CameraUniform mCameraUniform;

    glm::mat4 mModel;

    float mMouseX;
    float mMouseY;

    int mActiveDrawModeIndex;
    bool mWireframeOn;
    bool mResetModelMatrix;

  public:
    MeshDrawer();
    ~MeshDrawer();

    virtual void render(Clipboard &clipboard, Guid id) override;
};
} // namespace PhysicsEditor

#endif