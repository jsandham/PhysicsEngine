#ifndef __RIGIDBODY_DRAWER_H__
#define __RIGIDBODY_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class RigidbodyDrawer : public InspectorDrawer
{
  public:
    RigidbodyDrawer();
    ~RigidbodyDrawer();

    virtual void render(Clipboard& clipboard, Guid id) override;
};
} // namespace PhysicsEditor

#endif