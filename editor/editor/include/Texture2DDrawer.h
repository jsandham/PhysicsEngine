#ifndef __TEXTURE2D_DRAWER_H__
#define __TEXTURE2D_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class Texture2DDrawer : public InspectorDrawer
{
  public:
    Texture2DDrawer();
    ~Texture2DDrawer();

    void render(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif