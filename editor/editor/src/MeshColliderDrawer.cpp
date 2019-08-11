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

void MeshColliderDrawer::render(Component* component)
{
	if (ImGui::TreeNode("MeshCollider")) {
		MeshCollider* meshCollider = dynamic_cast<MeshCollider*>(component);

		ImGui::TreePop();
	}
}