#ifndef __TEXTURE3D_DRAWER_H__
#define __TEXTURE3D_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class Texture3DDrawer : public InspectorDrawer
{
  public:
    Texture3DDrawer();
    ~Texture3DDrawer();

    virtual void render(Clipboard &clipboard, Guid id) override;
};
} // namespace PhysicsEditor

#endif