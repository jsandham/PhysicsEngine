#include "../../include/drawers/CubemapDrawer.h"

#include "core/Cubemap.h"

#include "imgui.h"

using namespace PhysicsEditor;

CubemapDrawer::CubemapDrawer()
{
}

CubemapDrawer::~CubemapDrawer()
{
}

void CubemapDrawer::render(Clipboard &clipboard, Guid id)
{
	InspectorDrawer::render(clipboard, id);

	ImGui::Separator();
	mContentMin = ImGui::GetItemRectMin();

	ImGui::Separator();
	mContentMax = ImGui::GetItemRectMax();
}