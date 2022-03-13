#ifndef LINERENDERER_DRAWER_H__
#define LINERENDERER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class LineRendererDrawer : public InspectorDrawer
{
  public:
    LineRendererDrawer();
    ~LineRendererDrawer();

    virtual void render(Clipboard &clipboard, const Guid& id) override;
};
} // namespace PhysicsEditor

#endif