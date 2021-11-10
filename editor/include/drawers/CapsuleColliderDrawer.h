#ifndef __CAPSULECOLLIDER_DRAWER_H__
#define __CAPSULECOLLIDER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class CapsuleColliderDrawer : public InspectorDrawer
{
  public:
    CapsuleColliderDrawer();
    ~CapsuleColliderDrawer();

    virtual void render(Clipboard &clipboard, Guid id) override;
};
} // namespace PhysicsEditor

#endif