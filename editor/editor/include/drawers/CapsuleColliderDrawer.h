#ifndef __CAPSULECOLLIDER_DRAWER_H__
#define __CAPSULECOLLIDER_DRAWER_H__

#include "InspectorDrawer.h"
#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class CapsuleColliderDrawer : public InspectorDrawer
{
  public:
    CapsuleColliderDrawer();
    ~CapsuleColliderDrawer();

    void render(EditorClipboard& clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif