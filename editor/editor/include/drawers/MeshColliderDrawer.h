#ifndef __MESHCOLLIDER_DRAWER_H__
#define __MESHCOLLIDER_DRAWER_H__

#include "../EditorClipboard.h"
#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class MeshColliderDrawer : public InspectorDrawer
{
  public:
    MeshColliderDrawer();
    ~MeshColliderDrawer();

    void render(EditorClipboard &clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif