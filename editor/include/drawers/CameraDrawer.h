#ifndef CAMERA_DRAWER_H__
#define CAMERA_DRAWER_H__

#include <imgui.h>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class CameraDrawer
{
private:
    ImVec2 mContentMin;
    ImVec2 mContentMax;

  public:
    CameraDrawer();
    ~CameraDrawer();

    void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

private:
    bool isHovered() const;
};
} // namespace PhysicsEditor

#endif