#ifndef __MESHCOLLIDER_DRAWER_H__
#define __MESHCOLLIDER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class MeshColliderDrawer : public InspectorDrawer
{
  public:
    MeshColliderDrawer();
    ~MeshColliderDrawer();

    void render(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif