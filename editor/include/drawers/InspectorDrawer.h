#ifndef INSPECTOR_DRAWER_H__
#define INSPECTOR_DRAWER_H__

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
    InspectorDrawer(const InspectorDrawer&) = delete;
    InspectorDrawer& operator=(const InspectorDrawer&) = delete;
    virtual ~InspectorDrawer() = 0;

    virtual void render(Clipboard &clipboard, const Guid& id);

    ImVec2 getContentMin() const;
    ImVec2 getContentMax() const;

    bool isHovered() const;
};
} // namespace PhysicsEditor

#endif