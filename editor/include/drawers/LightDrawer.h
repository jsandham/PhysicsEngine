#ifndef __LIGHT_DRAWER_H__
#define __LIGHT_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class LightDrawer : public InspectorDrawer
{
  public:
    LightDrawer();
    ~LightDrawer();

    virtual void render(Clipboard &clipboard, const Guid& id) override;
};
} // namespace PhysicsEditor

#endif