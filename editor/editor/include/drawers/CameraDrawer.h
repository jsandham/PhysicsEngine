#ifndef __CAMERA_DRAWER_H__
#define __CAMERA_DRAWER_H__

#include "InspectorDrawer.h"
#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class CameraDrawer : public InspectorDrawer
{
  public:
    CameraDrawer();
    ~CameraDrawer();

    void render(EditorClipboard& clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif