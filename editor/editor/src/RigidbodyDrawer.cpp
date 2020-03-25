#include "../include/RigidbodyDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

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

void RigidbodyDrawer::render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id)
{
	if (ImGui::TreeNodeEx("Rigidbody", ImGuiTreeNodeFlags_DefaultOpen))
	{
		Rigidbody* rigidbody = world->getComponentById<Rigidbody>(id);

		ImGui::Text(("EntityId: " + rigidbody->getEntityId().toString()).c_str());
		ImGui::Text(("ComponentId: " + id.toString()).c_str());

		bool useGravity = rigidbody->mUseGravity;
		float mass = rigidbody->mMass;
		float drag = rigidbody->mDrag;
		float angularDrag = rigidbody->mAngularDrag;

		if (ImGui::Checkbox("Use Gravity", &useGravity)) {
			CommandManager::addCommand(new ChangePropertyCommand<bool>(&rigidbody->mUseGravity, useGravity, &scene.isDirty));
		}

		if (ImGui::InputFloat("Mass", &mass)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&rigidbody->mMass, mass, &scene.isDirty));
		}

		if (ImGui::InputFloat("Drag", &drag)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&rigidbody->mDrag, drag, &scene.isDirty));
		}

		if (ImGui::InputFloat("Angular Drag", &angularDrag)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&rigidbody->mAngularDrag, angularDrag, &scene.isDirty));
		}

		ImGui::TreePop();
	}
}