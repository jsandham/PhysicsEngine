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

    void render(Clipboard &clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif