#ifndef __INSPECTOR_H__
#define __INSPECTOR_H__

#include <vector>

#include "EditorClipboard.h"
#include "EditorProject.h"
#include "EditorScene.h"
#include "InspectorDrawer.h"

#include "MaterialDrawer.h"
#include "ShaderDrawer.h"
#include "Texture2DDrawer.h"

#include "core/Entity.h"
#include "core/World.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
class Inspector
{
  private:
    MaterialDrawer materialDrawer;
    ShaderDrawer shaderDrawer;
    Texture2DDrawer texture2DDrawer;

  public:
    Inspector();
    ~Inspector();
    Inspector(const Inspector &other) = delete;
    Inspector &operator=(const Inspector &other) = delete;

    void render(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard,
                bool isOpenedThisFrame);

  private:
    void drawEntity(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard);
};
} // namespace PhysicsEditor

#endif