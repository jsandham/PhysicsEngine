#ifndef __MESH_DRAWER_H__
#define __MESH_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class MeshDrawer : public InspectorDrawer
{
  private:
    unsigned int mFBO;
    unsigned int mColor;
    unsigned int mDepth;

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

    virtual void render(Clipboard &clipboard, const Guid& id) override;
};
} // namespace PhysicsEditor

#endif