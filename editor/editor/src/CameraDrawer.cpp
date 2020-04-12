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
		Transform* transform = camera->getComponent<Transform>(world);

		ImGui::Text(("EntityId: " + camera->getEntityId().toString()).c_str());
		ImGui::Text(("ComponentId: " + id.toString()).c_str());

		int mode = static_cast<int>(camera->mMode);

		const char* modeNames[] = { "Main", "Secondary"};

		if(ImGui::Combo("Mode", &mode, modeNames, 2)) {
			CommandManager::addCommand(new ChangePropertyCommand<CameraMode>(&camera->mMode, static_cast<CameraMode>(mode), &scene.isDirty));
		}

		glm::vec3 position = transform->mPosition;// camera->mPosition;
		glm::vec3 front = transform->getForward();// camera->mFront;
		glm::vec3 up = transform->getUp();// camera->mUp;
		glm::vec4 backgroundColor = camera->mBackgroundColor;

		if (ImGui::InputFloat3("Position", glm::value_ptr(position))) {
			transform->mPosition = position;
			//CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&camera->mPosition, position, &scene.isDirty));
		}
		if (ImGui::InputFloat3("Front", glm::value_ptr(front))) {
			//CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&camera->mFront, front, &scene.isDirty));
		}
		if (ImGui::InputFloat3("Up", glm::value_ptr(up))) {
			//CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&camera->mUp, up, &scene.isDirty));
		}
		if (ImGui::ColorEdit4("Background Color", glm::value_ptr(backgroundColor))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec4>(&camera->mBackgroundColor, backgroundColor, &scene.isDirty));
		}

		int ssao = static_cast<int>(camera->mSSAO);

		const char* ssaoNames[] = { "On", "Off" };

		if (ImGui::Combo("SSAO", &ssao, ssaoNames, 2)) {
			CommandManager::addCommand(new ChangePropertyCommand<CameraSSAO>(&camera->mSSAO, static_cast<CameraSSAO>(ssao), &scene.isDirty));
		}

		if (ImGui::TreeNode("Viewport"))
		{
			int x = camera->mViewport.mX;
			int y = camera->mViewport.mY;
			int width = camera->mViewport.mWidth;
			int height = camera->mViewport.mHeight;

			if (ImGui::InputInt("x", &x)) {
				CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->mViewport.mX, x, &scene.isDirty));
			}
			if (ImGui::InputInt("y", &y)) {
				CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->mViewport.mY, y, &scene.isDirty));
			}
			if (ImGui::InputInt("Width", &width)) {
				CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->mViewport.mWidth, width, &scene.isDirty));
			}
			if (ImGui::InputInt("Height", &height)) {
				CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->mViewport.mHeight, height, &scene.isDirty));
			}

			ImGui::TreePop();
		}

		if (ImGui::TreeNode("Frustum"))
		{
			float fov = camera->mFrustum.mFov;
			float nearPlane = camera->mFrustum.mNearPlane;
			float farPlane = camera->mFrustum.mFarPlane;

			if (ImGui::InputFloat("Field of View", &fov)) {
				CommandManager::addCommand(new ChangePropertyCommand<float>(&camera->mFrustum.mFov, fov, &scene.isDirty));
			}
			if (ImGui::InputFloat("Near Plane", &nearPlane)) {
				CommandManager::addCommand(new ChangePropertyCommand<float>(&camera->mFrustum.mNearPlane, nearPlane, &scene.isDirty));
			}
			if (ImGui::InputFloat("Far Plane", &farPlane)) {
				CommandManager::addCommand(new ChangePropertyCommand<float>(&camera->mFrustum.mFarPlane, farPlane, &scene.isDirty));
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