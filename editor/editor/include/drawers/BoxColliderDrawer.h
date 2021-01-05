#ifndef __BOXCOLLIDER_DRAWER_H__
#define __BOXCOLLIDER_DRAWER_H__

#include "InspectorDrawer.h"
#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class BoxColliderDrawer : public InspectorDrawer
{
  public:
    BoxColliderDrawer();
    ~BoxColliderDrawer();

    void render(EditorClipboard& clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif