#include "../../include/drawers/FontDrawer.h"

#include "core/Font.h"

#include "imgui.h"

using namespace PhysicsEditor;

FontDrawer::FontDrawer()
{
}

FontDrawer::~FontDrawer()
{
}

void FontDrawer::render(Clipboard &clipboard, Guid id)
{
	InspectorDrawer::render(clipboard, id);

	ImGui::Separator();
	mContentMin = ImGui::GetItemRectMin();

	ImGui::Separator();
	mContentMax = ImGui::GetItemRectMax();
}