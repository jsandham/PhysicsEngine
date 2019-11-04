#include "../include/MeshRendererDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "components/MeshRenderer.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

MeshRendererDrawer::MeshRendererDrawer()
{

}

MeshRendererDrawer::~MeshRendererDrawer()
{

}

void MeshRendererDrawer::render(World world, Guid entityId, Guid componentId)
{
	if(ImGui::TreeNode("MeshRenderer"))
	{
		MeshRenderer* meshRenderer = world.getComponentById<MeshRenderer>(componentId);

		ImGui::Text(("EntityId: " + entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + componentId.toString()).c_str());

		//Guid meshId;
		//Guid materialIds[8];

		bool isStatic = meshRenderer->isStatic;

		if (ImGui::Checkbox("Is Static?", &isStatic)) {
			CommandManager::addCommand(new ChangePropertyCommand<bool>(&meshRenderer->isStatic, isStatic));
		}

		ImGui::TreePop();
	}
}