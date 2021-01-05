#include "../../include/drawers/FontDrawer.h"
#include "../../include/CommandManager.h"
#include "../../include/EditorCommands.h"

#include "core/Font.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

FontDrawer::FontDrawer()
{
}

FontDrawer::~FontDrawer()
{
}

void FontDrawer::render(EditorClipboard &clipboard, Guid id)
{
}