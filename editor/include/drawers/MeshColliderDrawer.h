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

    virtual void render(Clipboard &clipboard, Guid id) override;
};
} // namespace PhysicsEditor

#endif