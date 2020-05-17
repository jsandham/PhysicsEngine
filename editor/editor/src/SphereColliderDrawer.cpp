#include "../include/SphereColliderDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "components/SphereCollider.h"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

SphereColliderDrawer::SphereColliderDrawer()
{

}

SphereColliderDrawer::~SphereColliderDrawer()
{

}

void SphereColliderDrawer::render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id)
{
	if (ImGui::TreeNodeEx("SphereCollider", ImGuiTreeNodeFlags_DefaultOpen))
	{
		SphereCollider* sphereCollider = world->getComponentById<SphereCollider>(id);

		ImGui::Text(("EntityId: " + sphereCollider->getEntityId().toString()).c_str());
		ImGui::Text(("ComponentId: " + id.toString()).c_str());

		if (ImGui::TreeNode("Sphere")) {
			glm::vec3 centre = sphereCollider->mSphere.mCentre;
			float radius = sphereCollider->mSphere.mRadius;

			if (ImGui::InputFloat3("Centre", glm::value_ptr(centre))) {
				CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&sphereCollider->mSphere.mCentre, centre, &scene.isDirty));
			}
			if (ImGui::InputFloat("Radius", &radius)) {
				CommandManager::addCommand(new ChangePropertyCommand<float>(&sphereCollider->mSphere.mRadius, radius, &scene.isDirty));
			}

			ImGui::TreePop();
		}

		ImGui::TreePop();
	}
}