#include "../include/CapsuleColliderDrawer.h"

#include "components/CapsuleCollider.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

CapsuleColliderDrawer::CapsuleColliderDrawer()
{

}

CapsuleColliderDrawer::~CapsuleColliderDrawer()
{

}

void CapsuleColliderDrawer::render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id)
{
	if (ImGui::TreeNodeEx("CapsuleCollider", ImGuiTreeNodeFlags_DefaultOpen)) {
		CapsuleCollider* capsuleCollider = world->getComponentById<CapsuleCollider>(id);

		ImGui::Text(("EntityId: " + capsuleCollider->entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + id.toString()).c_str());

		if (ImGui::TreeNode("Capsule")) {
			float centre[3];
			centre[0] = capsuleCollider->capsule.centre.x;
			centre[1] = capsuleCollider->capsule.centre.y;
			centre[2] = capsuleCollider->capsule.centre.z;

			ImGui::InputFloat3("Centre", &centre[0]);
			ImGui::InputFloat("Radius", &capsuleCollider->capsule.radius);
			ImGui::InputFloat("Height", &capsuleCollider->capsule.height);

			capsuleCollider->capsule.centre.x = centre[0];
			capsuleCollider->capsule.centre.y = centre[1];
			capsuleCollider->capsule.centre.z = centre[2];

			ImGui::TreePop();
		}


		ImGui::TreePop();
	}
}