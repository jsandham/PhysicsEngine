#ifndef __SHADER_DRAWER_H__
#define __SHADER_DRAWER_H__

#include "../EditorClipboard.h"
#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class ShaderDrawer : public InspectorDrawer
{
  public:
    ShaderDrawer();
    ~ShaderDrawer();

    void render(Clipboard& clipboard, Guid id);
};
} // namespace PhysicsEditor

#endif