#include "../include/CameraDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "components/Camera.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

CameraDrawer::CameraDrawer()
{

}

CameraDrawer::~CameraDrawer()
{

}

void CameraDrawer::render(World* world, EditorUI& ui, Guid entityId, Guid componentId)
{
	if (ImGui::TreeNodeEx("Camera", ImGuiTreeNodeFlags_DefaultOpen))
	{
		Camera* camera = world->getComponentById<Camera>(componentId);

		ImGui::Text(("EntityId: " + entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + componentId.toString()).c_str());

		glm::vec3 position = camera->position;
		glm::vec3 front = camera->front;
		glm::vec3 up = camera->up;
		glm::vec4 backgroundColor = camera->backgroundColor;

		if (ImGui::InputFloat3("Position", glm::value_ptr(position))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&camera->position, position));
		}
		if (ImGui::InputFloat3("Front", glm::value_ptr(front))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&camera->front, front));
		}
		if (ImGui::InputFloat3("Up", glm::value_ptr(up))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&camera->up, up));
		}
		if (ImGui::ColorEdit4("Background Color", glm::value_ptr(backgroundColor))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec4>(&camera->backgroundColor, backgroundColor));
		}

		if (ImGui::TreeNode("Viewport"))
		{
			int x = camera->viewport.x;
			int y = camera->viewport.y;
			int width = camera->viewport.width;
			int height = camera->viewport.height;

			if (ImGui::InputInt("x", &x)) {
				CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->viewport.x, x));
			}
			if (ImGui::InputInt("y", &y)) {
				CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->viewport.y, y));
			}
			if (ImGui::InputInt("Width", &width)) {
				CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->viewport.width, width));
			}
			if (ImGui::InputInt("Height", &height)) {
				CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->viewport.height, height));
			}

			ImGui::TreePop();
		}

		if (ImGui::TreeNode("Frustum"))
		{
			float fov = camera->frustum.fov;
			float nearPlane = camera->frustum.nearPlane;
			float farPlane = camera->frustum.farPlane;

			if (ImGui::InputFloat("Field of View", &fov)) {
				CommandManager::addCommand(new ChangePropertyCommand<float>(&camera->frustum.fov, fov));
			}
			if (ImGui::InputFloat("Near Plane", &nearPlane)) {
				CommandManager::addCommand(new ChangePropertyCommand<float>(&camera->frustum.nearPlane, nearPlane));
			}
			if (ImGui::InputFloat("Far Plane", &farPlane)) {
				CommandManager::addCommand(new ChangePropertyCommand<float>(&camera->frustum.farPlane, farPlane));
			}

			ImGui::TreePop();
		}

		ImGui::TreePop();
	}
}