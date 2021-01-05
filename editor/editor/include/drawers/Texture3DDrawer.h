#ifndef __TEXTURE3D_DRAWER_H__
#define __TEXTURE3D_DRAWER_H__

#include "InspectorDrawer.h"
#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class Texture3DDrawer : public InspectorDrawer
{
  public:
    Texture3DDrawer();
    ~Texture3DDrawer();

    void render(EditorClipboard& clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif