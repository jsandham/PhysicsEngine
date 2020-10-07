#include "../include/CubemapDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "core/Cubemap.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

CubemapDrawer::CubemapDrawer()
{
}

CubemapDrawer::~CubemapDrawer()
{
}

void CubemapDrawer::render(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard,
                           Guid id)
{
}