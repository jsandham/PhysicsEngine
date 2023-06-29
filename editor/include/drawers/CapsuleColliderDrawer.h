#ifndef CAPSULECOLLIDER_DRAWER_H__
#define CAPSULECOLLIDER_DRAWER_H__

#include <imgui.h>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class CapsuleColliderDrawer
{
private:
    ImVec2 mContentMin;
    ImVec2 mContentMax;

  public:
    CapsuleColliderDrawer();
    ~CapsuleColliderDrawer();

    void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

private:
    bool isHovered() const;
};
} // namespace PhysicsEditor

#endif