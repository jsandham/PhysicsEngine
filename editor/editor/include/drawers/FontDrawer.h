#ifndef __FONT_DRAWER_H__
#define __FONT_DRAWER_H__

#include "InspectorDrawer.h"
#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class FontDrawer : public InspectorDrawer
{
  public:
    FontDrawer();
    ~FontDrawer();

    void render(EditorClipboard& clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif