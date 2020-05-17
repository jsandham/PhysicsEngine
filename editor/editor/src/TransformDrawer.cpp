#include <math.h>

#include "../include/TransformDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "components/Transform.h"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

TransformDrawer::TransformDrawer()
{

}

TransformDrawer::~TransformDrawer()
{

}

void TransformDrawer::render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id)
{
	if (ImGui::TreeNodeEx("Transform", ImGuiTreeNodeFlags_DefaultOpen))
	{
		Transform* transform = world->getComponentById<Transform>(id);

		ImGui::Text(("EntityId: " + transform->getEntityId().toString()).c_str());
		ImGui::Text(("ComponentId: " + id.toString()).c_str());

		glm::vec3 position = transform->mPosition;
		glm::quat rotation = transform->mRotation;
		glm::vec3 scale = transform->mScale;

		//glm::vec3 eulerRotDeg = glm::degrees(glm::eulerAngles(rotation));

		if (ImGui::InputFloat3("Position", glm::value_ptr(position))){
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&transform->mPosition, position, &scene.isDirty));
		}
		if (ImGui::InputFloat4("Rotation", glm::value_ptr(rotation))) {

			//eulerRotDeg = glm::vec3(fmod(eulerRotDeg.x, 360.0f), fmod(eulerRotDeg.y, 360.0f), fmod(eulerRotDeg.z, 360.0f));

			//rotation = glm::quat(glm::radians(eulerRotDeg));

			CommandManager::addCommand(new ChangePropertyCommand<glm::quat>(&transform->mRotation, rotation, &scene.isDirty));
		}
		if (ImGui::InputFloat3("Scale", glm::value_ptr(scale))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&transform->mScale, scale, &scene.isDirty));
		}

		ImGui::TreePop();
	}
}