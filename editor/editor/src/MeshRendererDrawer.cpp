#include "../include/MeshRendererDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "components/MeshRenderer.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

#include "../include/imgui_extensions.h"

using namespace PhysicsEditor;

MeshRendererDrawer::MeshRendererDrawer()
{

}

MeshRendererDrawer::~MeshRendererDrawer()
{

}

void MeshRendererDrawer::render(World* world, EditorUI& ui, Guid entityId, Guid componentId)
{
	if(ImGui::TreeNodeEx("MeshRenderer", ImGuiTreeNodeFlags_DefaultOpen))
	{
		MeshRenderer* meshRenderer = world->getComponentById<MeshRenderer>(componentId);

		ImGui::Text(("EntityId: " + entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + componentId.toString()).c_str());

		// Mesh
		Guid meshId = meshRenderer->meshId;
		std::string meshName = "None (Mesh)";
		if (meshId != Guid::INVALID) {
			meshName = meshId.toString();
		}

		bool slotFillable = ui.draggedId != PhysicsEngine::Guid::INVALID && world->getAsset<Mesh>(ui.draggedId) != NULL;
		bool slotFilled = false;
		bool isClicked = ImGui::Slot("Mesh", meshName, slotFillable, &slotFilled);

		if (slotFilled) {
			// TODO: Need some way of telling the renderer that the mesh has changed. 
			meshId = ui.draggedId;
			ui.draggedId = PhysicsEngine::Guid::INVALID;

			CommandManager::addCommand(new ChangePropertyCommand<Guid>(&meshRenderer->meshId, meshId));
		}

		bool isStatic = meshRenderer->isStatic;
		if (ImGui::Checkbox("Is Static?", &isStatic)) {
			CommandManager::addCommand(new ChangePropertyCommand<bool>(&meshRenderer->isStatic, isStatic));
		}






		// Materials
		static int materialCount = 1;
		static int increment = 1;
		ImGui::PushItemWidth(80);
		ImGui::InputScalar("Material Count", ImGuiDataType_S32, &materialCount, &increment, NULL, "%d");
		ImGui::PopItemWidth();

		materialCount = std::max(0, std::min(materialCount, 8));

		// TODO: We probably need to add an int variable to the material class to store how many active materials there are.


		Guid materialIds[8];
		//int activeMaterialsCount = 0;
		for (int i = 0; i < materialCount; i++) {
			materialIds[i] = meshRenderer->materialIds[i];

			std::string materialName = "None (Material)";
			if (materialIds[i] != PhysicsEngine::Guid::INVALID) {
				materialName = materialIds[i].toString();
				//activeMaterialsCount++;
			}

			bool materialSlotFillable = ui.draggedId != PhysicsEngine::Guid::INVALID && world->getAsset<Material>(ui.draggedId) != NULL;
			bool materialSlotFilled = false;
			bool materialIsClicked = ImGui::Slot("Material", materialName, materialSlotFillable, &materialSlotFilled);

			if (materialSlotFilled) {
				materialIds[i] = ui.draggedId;
				ui.draggedId = PhysicsEngine::Guid::INVALID;
			}
		}

		ImGui::TreePop();
	}
}