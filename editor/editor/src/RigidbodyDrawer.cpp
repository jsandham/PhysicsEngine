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

void RigidbodyDrawer::render(World* world, EditorUI& ui, Guid entityId, Guid componentId)
{
	if (ImGui::TreeNodeEx("Rigidbody", ImGuiTreeNodeFlags_DefaultOpen))
	{
		Rigidbody* rigidbody = world->getComponentById<Rigidbody>(componentId);

		ImGui::Text(("EntityId: " + entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + componentId.toString()).c_str());

		bool useGravity = rigidbody->useGravity;
		float mass = rigidbody->mass;
		float drag = rigidbody->drag;
		float angularDrag = rigidbody->angularDrag;

		if (ImGui::Checkbox("Use Gravity", &useGravity)) {
			CommandManager::addCommand(new ChangePropertyCommand<bool>(&rigidbody->useGravity, useGravity));
		}

		if (ImGui::InputFloat("Mass", &mass)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&rigidbody->mass, mass));
		}

		if (ImGui::InputFloat("Drag", &drag)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&rigidbody->drag, drag));
		}

		if (ImGui::InputFloat("Angular Drag", &angularDrag)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&rigidbody->angularDrag, angularDrag));
		}

		ImGui::TreePop();
	}
}