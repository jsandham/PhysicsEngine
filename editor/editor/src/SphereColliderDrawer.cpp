#include "../include/SphereColliderDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "components/SphereCollider.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

SphereColliderDrawer::SphereColliderDrawer()
{

}

SphereColliderDrawer::~SphereColliderDrawer()
{

}

void SphereColliderDrawer::render(World* world, EditorUI& ui, Guid entityId, Guid componentId)
{
	if (ImGui::TreeNodeEx("SphereCollider", ImGuiTreeNodeFlags_DefaultOpen))
	{
		SphereCollider* sphereCollider = world->getComponentById<SphereCollider>(componentId);

		ImGui::Text(("EntityId: " + entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + componentId.toString()).c_str());

		if (ImGui::TreeNode("Sphere")) {
			glm::vec3 centre = sphereCollider->sphere.centre;
			float radius = sphereCollider->sphere.radius;

			if (ImGui::InputFloat3("Centre", glm::value_ptr(centre))) {
				CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&sphereCollider->sphere.centre, centre));
			}
			if (ImGui::InputFloat("Radius", &radius)) {
				CommandManager::addCommand(new ChangePropertyCommand<float>(&sphereCollider->sphere.radius, radius));
			}

			ImGui::TreePop();
		}

		ImGui::TreePop();
	}
}