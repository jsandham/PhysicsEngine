#ifndef __LIGHT_DRAWER_H__
#define __LIGHT_DRAWER_H__

#include "InspectorDrawer.h"
#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class LightDrawer : public InspectorDrawer
{
  public:
    LightDrawer();
    ~LightDrawer();

    void render(EditorClipboard& clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif