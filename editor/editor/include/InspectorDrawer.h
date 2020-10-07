#ifndef __INSPECTOR_DRAWER_H__
#define __INSPECTOR_DRAWER_H__

#include "core/Guid.h"
#include "core/World.h"

#include "EditorClipboard.h"
#include "EditorProject.h"
#include "EditorScene.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
class InspectorDrawer
{
  public:
    InspectorDrawer();
    virtual ~InspectorDrawer() = 0;

    // virtual void init(World* world);
    virtual void render(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard,
                        Guid id) = 0;
};
} // namespace PhysicsEditor

#endif