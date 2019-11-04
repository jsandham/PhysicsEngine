#include "../include/TransformDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "components/Transform.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

TransformDrawer::TransformDrawer()
{

}

TransformDrawer::~TransformDrawer()
{

}

void TransformDrawer::render(World world, Guid entityId, Guid componentId)
{
	if (ImGui::TreeNode("Transform"))
	{
		Transform* transform = world.getComponentById<Transform>(componentId);

		ImGui::Text(("EntityId: " + entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + componentId.toString()).c_str());

		glm::vec3 position = transform->position;
		glm::quat rotation = transform->rotation;
		glm::vec3 scale = transform->scale;

		if (ImGui::InputFloat3("Position", glm::value_ptr(position))){
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&transform->position, position));
		}
		if (ImGui::InputFloat4("Rotation", glm::value_ptr(rotation))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::quat>(&transform->rotation, rotation));
		}
		if (ImGui::InputFloat3("Scale", glm::value_ptr(scale))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&transform->scale, scale));
		}

		ImGui::TreePop();
	}
}