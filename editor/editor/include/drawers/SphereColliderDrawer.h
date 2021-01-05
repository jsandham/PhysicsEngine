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

    void render(EditorClipboard &clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif