#include "../include/BoxColliderDrawer.h"

#include "components/BoxCollider.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

BoxColliderDrawer::BoxColliderDrawer()
{

}

BoxColliderDrawer::~BoxColliderDrawer()
{

}

void BoxColliderDrawer::render(World world, Guid entityId, Guid componentId)
{
	if (ImGui::TreeNode("BoxCollider")) {
		BoxCollider* boxCollider = world.getComponentById<BoxCollider>(componentId);

		ImGui::Text(("EntityId: " + entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + componentId.toString()).c_str());

		if (ImGui::TreeNode("Bounds")) {
			float centre[3];
			centre[0] = boxCollider->bounds.centre.x;
			centre[1] = boxCollider->bounds.centre.y;
			centre[2] = boxCollider->bounds.centre.z;

			float size[3];
			size[0] = boxCollider->bounds.size.x;
			size[1] = boxCollider->bounds.size.y;
			size[2] = boxCollider->bounds.size.z;

			ImGui::InputFloat3("Centre", &centre[0]);
			ImGui::InputFloat3("Size", &size[0]);

			boxCollider->bounds.centre.x = centre[0];
			boxCollider->bounds.centre.y = centre[1];
			boxCollider->bounds.centre.z = centre[2];

			boxCollider->bounds.size.x = size[0];
			boxCollider->bounds.size.y = size[1];
			boxCollider->bounds.size.z = size[2];

			ImGui::TreePop();
		}


		ImGui::TreePop();
	}
}