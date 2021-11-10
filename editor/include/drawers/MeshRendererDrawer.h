#ifndef __MESHRENDERER_DRAWER_H__
#define __MESHRENDERER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class MeshRendererDrawer : public InspectorDrawer
{
  public:
    MeshRendererDrawer();
    ~MeshRendererDrawer();

    virtual void render(Clipboard &clipboard, Guid id) override;
};
} // namespace PhysicsEditor

#endif