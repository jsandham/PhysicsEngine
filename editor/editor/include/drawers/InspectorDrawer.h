#ifndef __INSPECTOR_DRAWER_H__
#define __INSPECTOR_DRAWER_H__

#include "core/Guid.h"
#include "core/World.h"

#include "../EditorClipboard.h"

#include "imgui.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
class InspectorDrawer
{
  protected:
      ImVec2 mContentMin;
      ImVec2 mContentMax;
  
  public:
    InspectorDrawer();
    virtual ~InspectorDrawer() = 0;

    virtual void render(Clipboard &clipboard, Guid id);

    ImVec2 getContentMin() const;
    ImVec2 getContentMax() const;

    bool isHovered() const;
};
} // namespace PhysicsEditor

#endif