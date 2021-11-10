#ifndef __CUBEMAP_DRAWER_H__
#define __CUBEMAP_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class CubemapDrawer : public InspectorDrawer
{
  public:
    CubemapDrawer();
    ~CubemapDrawer();

    virtual void render(Clipboard &clipboard, Guid id) override;
};
} // namespace PhysicsEditor

#endif