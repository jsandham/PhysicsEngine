#include "../../include/drawers/Texture3DDrawer.h"
#include "../../include/EditorClipboard.h"

#include "core/Texture3D.h"

#include "imgui.h"

using namespace PhysicsEditor;

Texture3DDrawer::Texture3DDrawer()
{
}

Texture3DDrawer::~Texture3DDrawer()
{
}

void Texture3DDrawer::render(Clipboard &clipboard, const Guid& id)
{
	InspectorDrawer::render(clipboard, id);

	ImGui::Separator();
	mContentMin = ImGui::GetItemRectMin();

	ImGui::Separator();
	mContentMax = ImGui::GetItemRectMax();
}