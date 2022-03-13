#ifndef CAPSULECOLLIDER_DRAWER_H__
#define CAPSULECOLLIDER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class CapsuleColliderDrawer : public InspectorDrawer
{
  public:
    CapsuleColliderDrawer();
    ~CapsuleColliderDrawer();

    virtual void render(Clipboard &clipboard, const Guid& id) override;
};
} // namespace PhysicsEditor

#endif