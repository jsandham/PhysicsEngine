#include "../include/MeshColliderDrawer.h"

#include "components/MeshCollider.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

MeshColliderDrawer::MeshColliderDrawer()
{

}

MeshColliderDrawer::~MeshColliderDrawer()
{

}

void MeshColliderDrawer::render(World world, Guid entityId, Guid componentId)
{
	if (ImGui::TreeNode("MeshCollider")) {
		MeshCollider* meshCollider = world.getComponentById<MeshCollider>(componentId);

		ImGui::Text(("EntityId: " + entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + componentId.toString()).c_str());

		ImGui::TreePop();
	}
}