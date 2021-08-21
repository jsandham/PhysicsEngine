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

    virtual void render(Clipboard& clipboard, Guid id) override;
};
} // namespace PhysicsEditor

#endif