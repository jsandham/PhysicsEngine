#include "../../include/drawers/CubemapDrawer.h"
#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"

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

void CubemapDrawer::render(EditorClipboard &clipboard, Guid id)
{
}