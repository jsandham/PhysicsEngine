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

    virtual void render(Clipboard &clipboard, const Guid& id) override;
};
} // namespace PhysicsEditor

#endif