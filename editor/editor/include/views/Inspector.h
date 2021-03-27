#ifndef __INSPECTOR_H__
#define __INSPECTOR_H__

#include <vector>

#include "Window.h"

#include "../drawers/InspectorDrawer.h"
#include "../drawers/MaterialDrawer.h"
#include "../drawers/MeshDrawer.h"
#include "../drawers/ShaderDrawer.h"
#include "../drawers/Texture2DDrawer.h"

namespace PhysicsEditor
{
class Inspector : public Window
{
  private:
    MeshDrawer meshDrawer;
    MaterialDrawer materialDrawer;
    ShaderDrawer shaderDrawer;
    Texture2DDrawer texture2DDrawer;

  public:
    Inspector();
    ~Inspector();
    Inspector(const Inspector &other) = delete;
    Inspector &operator=(const Inspector &other) = delete;

    void init(Clipboard &clipboard) override;
    void update(Clipboard &clipboard) override;

  private:
    void drawEntity(Clipboard &clipboard);
};
} // namespace PhysicsEditor

#endif