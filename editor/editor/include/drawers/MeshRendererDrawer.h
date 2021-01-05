#ifndef __MESHRENDERER_DRAWER_H__
#define __MESHRENDERER_DRAWER_H__

#include "InspectorDrawer.h"
#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class MeshRendererDrawer : public InspectorDrawer
{
  public:
    MeshRendererDrawer();
    ~MeshRendererDrawer();

    void render(EditorClipboard& clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif