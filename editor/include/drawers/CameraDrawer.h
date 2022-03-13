#ifndef CAMERA_DRAWER_H__
#define CAMERA_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class CameraDrawer : public InspectorDrawer
{
  public:
    CameraDrawer();
    ~CameraDrawer();

    virtual void render(Clipboard &clipboard, const Guid& id) override;
};
} // namespace PhysicsEditor

#endif