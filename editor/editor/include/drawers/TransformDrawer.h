#ifndef __TRANSFORM_DRAWER_H__
#define __TRANSFORM_DRAWER_H__

#include "../EditorClipboard.h"
#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class TransformDrawer : public InspectorDrawer
{
  public:
    TransformDrawer();
    ~TransformDrawer();

    void render(EditorClipboard &clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif