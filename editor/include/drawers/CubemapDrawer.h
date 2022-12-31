#ifndef CUBEMAP_DRAWER_H__
#define CUBEMAP_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class CubemapDrawer : public InspectorDrawer
{
  public:
    CubemapDrawer();
    ~CubemapDrawer();

    virtual void render(Clipboard &clipboard, const Guid& id) override;

  private:
    void drawCubemapFaceTexture(Clipboard& clipboard, CubemapFace face, Cubemap* cubemap, Texture2D* texture);
};
} // namespace PhysicsEditor

#endif