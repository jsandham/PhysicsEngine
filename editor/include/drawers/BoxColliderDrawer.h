#ifndef BOXCOLLIDER_DRAWER_H__
#define BOXCOLLIDER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class BoxColliderDrawer : public InspectorDrawer
{
  public:
    BoxColliderDrawer();
    ~BoxColliderDrawer();

    virtual void render(Clipboard &clipboard, const Guid& id) override;
};
} // namespace PhysicsEditor

#endif