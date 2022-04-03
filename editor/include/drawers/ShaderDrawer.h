#ifndef SHADER_DRAWER_H__
#define SHADER_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class ShaderDrawer : public InspectorDrawer
{
  private:
      PhysicsEngine::Guid mShaderId;
  public:
    ShaderDrawer();
    ~ShaderDrawer();

    virtual void render(Clipboard& clipboard, const Guid& id) override;
};
} // namespace PhysicsEditor

#endif