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

    void render(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif