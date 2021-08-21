#ifndef __CAMERA_DRAWER_H__
#define __CAMERA_DRAWER_H__

#include "../EditorClipboard.h"
#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class CameraDrawer : public InspectorDrawer
{
  public:
    CameraDrawer();
    ~CameraDrawer();

    virtual void render(Clipboard &clipboard, Guid id) override;
};
} // namespace PhysicsEditor

#endif