#ifndef __SPHERECOLLIDER_DRAWER_H__
#define __SPHERECOLLIDER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class SphereColliderDrawer : public InspectorDrawer
{
  public:
    SphereColliderDrawer();
    ~SphereColliderDrawer();

    void render(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif