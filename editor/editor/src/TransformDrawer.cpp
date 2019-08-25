#include "../include/TransformDrawer.h"

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

		float position[3];
		position[0] = transform->position.x;
		position[1] = transform->position.y;
		position[2] = transform->position.z;

		float rotation[4];
		rotation[0] = transform->rotation.x;
		rotation[1] = transform->rotation.y;
		rotation[2] = transform->rotation.z;
		rotation[3] = transform->rotation.w;

		float scale[3];
		scale[0] = transform->scale.x;
		scale[1] = transform->scale.y;
		scale[2] = transform->scale.z;

		ImGui::InputFloat3("Position", &position[0]);
		ImGui::InputFloat4("Rotation", &rotation[0]);
		ImGui::InputFloat3("Scale", &scale[0]);

		transform->position.x = position[0];
		transform->position.y = position[1];
		transform->position.z = position[2];

		transform->rotation.x = rotation[0];
		transform->rotation.y = rotation[1];
		transform->rotation.z = rotation[2];
		transform->rotation.w = rotation[3];

		transform->scale.x = scale[0];
		transform->scale.y = scale[1];
		transform->scale.z = scale[2];

		ImGui::TreePop();
	}
}