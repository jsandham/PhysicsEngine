#include "../../include/drawers/InspectorDrawer.h"

#include "imgui.h"

#include "core/Rect.h"

using namespace PhysicsEditor;

InspectorDrawer::InspectorDrawer() : mContentMin(0, 0), mContentMax(0, 0)
{
}

InspectorDrawer::~InspectorDrawer()
{
}

void InspectorDrawer::render(Clipboard& clipboard, const Guid& id)
{
	if (isHovered())
	{
		ImGui::GetForegroundDrawList()->AddRect(mContentMin, mContentMax, 0xFF00FF00);
	}
}

ImVec2 InspectorDrawer::getContentMin() const
{
	return mContentMin;
}

ImVec2 InspectorDrawer::getContentMax() const
{
	return mContentMax;
}

bool InspectorDrawer::isHovered() const
{
	ImVec2 cursorPos = ImGui::GetMousePos();

	ImGui::GetForegroundDrawList()->AddCircle(cursorPos, 4.0f, 0xFF00FF00);

	glm::vec2 min = glm::vec2(mContentMin.x, mContentMin.y);
	glm::vec2 max = glm::vec2(mContentMax.x, mContentMax.y);

	PhysicsEngine::Rect rect(min, max);

	return rect.contains(cursorPos.x, cursorPos.y);
}