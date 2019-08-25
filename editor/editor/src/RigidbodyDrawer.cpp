#include "../include/RigidbodyDrawer.h"

#include "components/Rigidbody.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

RigidbodyDrawer::RigidbodyDrawer()
{

}

RigidbodyDrawer::~RigidbodyDrawer()
{

}

void RigidbodyDrawer::render(World world, Guid entityId, Guid componentId)
{
	if (ImGui::TreeNode("Rigidbody"))
	{
		Rigidbody* rigidbody = world.getComponentById<Rigidbody>(componentId);

		ImGui::Text(("EntityId: " + entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + componentId.toString()).c_str());

		ImGui::Checkbox("Use Gravity", &rigidbody->useGravity);
		ImGui::InputFloat("Mass", &rigidbody->mass);
		ImGui::InputFloat("Drag", &rigidbody->drag);
		ImGui::InputFloat("Angular Drag", &rigidbody->angularDrag);

		ImGui::TreePop();
	}
}