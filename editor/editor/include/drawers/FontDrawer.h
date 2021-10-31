#ifndef __FONT_DRAWER_H__
#define __FONT_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class FontDrawer : public InspectorDrawer
{
  public:
    FontDrawer();
    ~FontDrawer();

    virtual void render(Clipboard &clipboard, Guid id) override;
};
} // namespace PhysicsEditor

#endif