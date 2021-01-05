#ifndef __CUBEMAP_DRAWER_H__
#define __CUBEMAP_DRAWER_H__

#include "../EditorClipboard.h"
#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class CubemapDrawer : public InspectorDrawer
{
  public:
    CubemapDrawer();
    ~CubemapDrawer();

    void render(EditorClipboard &clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif