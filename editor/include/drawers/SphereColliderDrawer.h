#ifndef SPHERECOLLIDER_DRAWER_H__
#define SPHERECOLLIDER_DRAWER_H__

#include <imgui.h>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class SphereColliderDrawer
{
  private:
    ImVec2 mContentMin;
    ImVec2 mContentMax;

  public:
    SphereColliderDrawer();
    ~SphereColliderDrawer();

    void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

private:
    bool isHovered() const;
};
} // namespace PhysicsEditor

#endif