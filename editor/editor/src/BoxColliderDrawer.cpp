#include "../include/BoxColliderDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "components/BoxCollider.h"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

BoxColliderDrawer::BoxColliderDrawer()
{

}

BoxColliderDrawer::~BoxColliderDrawer()
{

}

void BoxColliderDrawer::render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id)
{
	if (ImGui::TreeNodeEx("BoxCollider", ImGuiTreeNodeFlags_DefaultOpen)) {
		BoxCollider* boxCollider = world->getComponentById<BoxCollider>(id);

		ImGui::Text(("EntityId: " + boxCollider->getEntityId().toString()).c_str());
		ImGui::Text(("ComponentId: " + id.toString()).c_str());

		if (ImGui::TreeNode("Bounds")) {
			glm::vec3 centre = boxCollider->mAABB.mCentre;
			glm::vec3 size = boxCollider->mAABB.mSize;

			if (ImGui::InputFloat3("Centre", glm::value_ptr(centre))) {
				CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&boxCollider->mAABB.mCentre, centre, &scene.isDirty));
			}
			if (ImGui::InputFloat3("Size", glm::value_ptr(size))) {
				CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&boxCollider->mAABB.mSize, size, &scene.isDirty));
			}

			ImGui::TreePop();
		}


		ImGui::TreePop();
	}
}