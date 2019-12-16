#include "../include/LineRendererDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "components/LineRenderer.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

LineRendererDrawer::LineRendererDrawer()
{

}

LineRendererDrawer::~LineRendererDrawer()
{

}

void LineRendererDrawer::render(World* world, EditorClipboard& clipboard, Guid id)
{
	if (ImGui::TreeNodeEx("LineRenderer", ImGuiTreeNodeFlags_DefaultOpen)) {
		LineRenderer* lineRenderer = world->getComponentById<LineRenderer>(id);

		ImGui::Text(("EntityId: " + lineRenderer->entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + id.toString()).c_str());

		glm::vec3 start = lineRenderer->start;
		glm::vec3 end = lineRenderer->end;

		if (ImGui::InputFloat3("Start", glm::value_ptr(start))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&lineRenderer->start, start));
		}
		if (ImGui::InputFloat3("End", glm::value_ptr(end))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&lineRenderer->end, end));
		}

		ImGui::TreePop();
	}
}