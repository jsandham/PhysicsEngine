#include "../include/Texture3DDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "core/Texture3D.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

Texture3DDrawer::Texture3DDrawer()
{
}

Texture3DDrawer::~Texture3DDrawer()
{
}

void Texture3DDrawer::render(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard,
                             Guid id)
{
}