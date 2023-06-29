#ifndef BOXCOLLIDER_DRAWER_H__
#define BOXCOLLIDER_DRAWER_H__

#include <imgui.h>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class BoxColliderDrawer
{
  private:
    ImVec2 mContentMin;
    ImVec2 mContentMax;
 
  public:
    BoxColliderDrawer();
    ~BoxColliderDrawer();

    void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

private:
    bool isHovered() const;
};
} // namespace PhysicsEditor

#endif