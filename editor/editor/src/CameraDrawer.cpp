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

void CameraDrawer::render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id)
{
	if (ImGui::TreeNodeEx("Camera", ImGuiTreeNodeFlags_DefaultOpen))
	{
		Camera* camera = world->getComponentById<Camera>(id);

		ImGui::Text(("EntityId: " + camera->getEntityId().toString()).c_str());
		ImGui::Text(("ComponentId: " + id.toString()).c_str());

		int mode = static_cast<int>(camera->mode);

		const char* modeNames[] = { "Main", "Secondary"};

		if(ImGui::Combo("Mode", &mode, modeNames, 2)) {
			CommandManager::addCommand(new ChangePropertyCommand<CameraMode>(&camera->mode, static_cast<CameraMode>(mode), &scene.isDirty));
		}

		glm::vec3 position = camera->position;
		glm::vec3 front = camera->front;
		glm::vec3 up = camera->up;
		glm::vec4 backgroundColor = camera->backgroundColor;

		if (ImGui::InputFloat3("Position", glm::value_ptr(position))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&camera->position, position, &scene.isDirty));
		}
		if (ImGui::InputFloat3("Front", glm::value_ptr(front))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&camera->front, front, &scene.isDirty));
		}
		if (ImGui::InputFloat3("Up", glm::value_ptr(up))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&camera->up, up, &scene.isDirty));
		}
		if (ImGui::ColorEdit4("Background Color", glm::value_ptr(backgroundColor))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec4>(&camera->backgroundColor, backgroundColor, &scene.isDirty));
		}

		bool useSSAO = camera->useSSAO;

		if(ImGui::Checkbox("SSAO", &useSSAO)){
			CommandManager::addCommand(new ChangePropertyCommand<bool>(&camera->useSSAO, useSSAO, &scene.isDirty));
		}

		if (ImGui::TreeNode("Viewport"))
		{
			int x = camera->viewport.x;
			int y = camera->viewport.y;
			int width = camera->viewport.width;
			int height = camera->viewport.height;

			if (ImGui::InputInt("x", &x)) {
				CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->viewport.x, x, &scene.isDirty));
			}
			if (ImGui::InputInt("y", &y)) {
				CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->viewport.y, y, &scene.isDirty));
			}
			if (ImGui::InputInt("Width", &width)) {
				CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->viewport.width, width, &scene.isDirty));
			}
			if (ImGui::InputInt("Height", &height)) {
				CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->viewport.height, height, &scene.isDirty));
			}

			ImGui::TreePop();
		}

		if (ImGui::TreeNode("Frustum"))
		{
			float fov = camera->frustum.fov;
			float nearPlane = camera->frustum.nearPlane;
			float farPlane = camera->frustum.farPlane;

			if (ImGui::InputFloat("Field of View", &fov)) {
				CommandManager::addCommand(new ChangePropertyCommand<float>(&camera->frustum.fov, fov, &scene.isDirty));
			}
			if (ImGui::InputFloat("Near Plane", &nearPlane)) {
				CommandManager::addCommand(new ChangePropertyCommand<float>(&camera->frustum.nearPlane, nearPlane, &scene.isDirty));
			}
			if (ImGui::InputFloat("Far Plane", &farPlane)) {
				CommandManager::addCommand(new ChangePropertyCommand<float>(&camera->frustum.farPlane, farPlane, &scene.isDirty));
			}

			ImGui::TreePop();
		}

		// if (ImGui::TreeNode("Targets")) {
		// 	bool useColor = camera->useColorTarget;
		// 	bool usePosition = camera->usePositionTarget;
		// 	bool useNormal = camera->useNormalTarget;
		// 	bool useDepth = camera->useDepthTarget;

		// 	if (ImGui::Checkbox("Color", &useColor)) {
		// 		CommandManager::addCommand(new ChangePropertyCommand<bool>(&camera->useColorTarget, useColor));
		// 	}

		// 	if (ImGui::Checkbox("Position", &usePosition)) {
		// 		CommandManager::addCommand(new ChangePropertyCommand<bool>(&camera->usePositionTarget, usePosition));
		// 	}

		// 	if (ImGui::Checkbox("Normal", &useNormal)) {
		// 		CommandManager::addCommand(new ChangePropertyCommand<bool>(&camera->useNormalTarget, useNormal));
		// 	}

		// 	if (ImGui::Checkbox("Depth", &useDepth)) {
		// 		CommandManager::addCommand(new ChangePropertyCommand<bool>(&camera->useDepthTarget, useDepth));
		// 	}

		// 	ImGui::TreePop();
		// }

		ImGui::TreePop();
	}
}