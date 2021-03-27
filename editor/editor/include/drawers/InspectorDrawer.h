#ifndef __INSPECTOR_DRAWER_H__
#define __INSPECTOR_DRAWER_H__

#include "core/Guid.h"
#include "core/World.h"

#include "../EditorClipboard.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
class InspectorDrawer
{
  public:
    InspectorDrawer();
    virtual ~InspectorDrawer() = 0;

    virtual void render(Clipboard &clipboard, Guid id) = 0;
};
} // namespace PhysicsEditor

#endif