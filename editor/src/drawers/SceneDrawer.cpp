#include "../../include/drawers/SceneDrawer.h"

using namespace PhysicsEditor;

SceneDrawer::SceneDrawer()
{

}

SceneDrawer::~SceneDrawer()
{

}

void SceneDrawer::render(Clipboard& clipboard, const PhysicsEngine::Guid& id)
{
	ImGui::Separator();
	mContentMin = ImGui::GetItemRectMin();

	if (ImGui::TreeNodeEx("Scene", ImGuiTreeNodeFlags_DefaultOpen))
	{
		PhysicsEngine::Scene* scene = clipboard.getWorld()->getSceneByGuid(id);

		if (scene != nullptr)
		{
			ImGui::Text("Hello World");
		}
		else
		{
			ImGui::Text("Goodbye World");
		}

		ImGui::TreePop();
	}

	ImGui::Separator();
	mContentMax = ImGui::GetItemRectMax();
}

bool SceneDrawer::isHovered() const
{
	ImVec2 cursorPos = ImGui::GetMousePos();

	glm::vec2 min = glm::vec2(mContentMin.x, mContentMin.y);
	glm::vec2 max = glm::vec2(mContentMax.x, mContentMax.y);

	PhysicsEngine::Rect rect(min, max);

	return rect.contains(cursorPos.x, cursorPos.y);
}