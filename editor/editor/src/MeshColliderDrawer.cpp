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

void MeshColliderDrawer::render(World* world, EditorClipboard& clipboard, Guid id)
{
	if (ImGui::TreeNodeEx("MeshCollider", ImGuiTreeNodeFlags_DefaultOpen)) {
		MeshCollider* meshCollider = world->getComponentById<MeshCollider>(id);

		ImGui::Text(("EntityId: " + meshCollider->entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + id.toString()).c_str());

		ImGui::TreePop();
	}
}