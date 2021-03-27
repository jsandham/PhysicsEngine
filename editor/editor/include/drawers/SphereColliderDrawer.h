#ifndef __SPHERECOLLIDER_DRAWER_H__
#define __SPHERECOLLIDER_DRAWER_H__

#include "../EditorClipboard.h"
#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class SphereColliderDrawer : public InspectorDrawer
{
  public:
    SphereColliderDrawer();
    ~SphereColliderDrawer();

    void render(Clipboard &clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif