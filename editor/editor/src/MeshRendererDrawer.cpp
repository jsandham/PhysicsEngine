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

void MeshRendererDrawer::render(World* world, EditorClipboard& clipboard, Guid id)
{
	if(ImGui::TreeNodeEx("MeshRenderer", ImGuiTreeNodeFlags_DefaultOpen))
	{
		MeshRenderer* meshRenderer = world->getComponentById<MeshRenderer>(id);

		ImGui::Text(("EntityId: " + meshRenderer->entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + id.toString()).c_str());

		// Mesh
		Guid meshId = meshRenderer->meshId;
		std::string meshName = "None (Mesh)";
		if (meshId != Guid::INVALID) {
			meshName = meshId.toString();
		}

		bool slotFillable = clipboard.getDraggedType() == InteractionType::Mesh;
		bool slotFilled = false;
		bool isClicked = ImGui::Slot("Mesh", meshName, slotFillable, &slotFilled);

		if (slotFilled) {
			// TODO: Need some way of telling the renderer that the mesh has changed. 
			meshId = clipboard.getDraggedId();
			clipboard.clearDraggedItem();

			CommandManager::addCommand(new ChangePropertyCommand<Guid>(&meshRenderer->meshId, meshId));
		}

		bool isStatic = meshRenderer->isStatic;
		if (ImGui::Checkbox("Is Static?", &isStatic)) {
			CommandManager::addCommand(new ChangePropertyCommand<bool>(&meshRenderer->isStatic, isStatic));
		}

		// Materials
		int materialCount = meshRenderer->materialCount;
		const int increment = 1;
		ImGui::PushItemWidth(80);
		if (ImGui::InputScalar("Material Count", ImGuiDataType_S32, &materialCount, &increment, NULL, "%d")) {
			materialCount = std::max(0, std::min(materialCount, 8));

			CommandManager::addCommand(new ChangePropertyCommand<int>(&meshRenderer->materialCount, materialCount));
		}
		ImGui::PopItemWidth();

		Guid materialIds[8];
		for (int i = 0; i < materialCount; i++) {
			materialIds[i] = meshRenderer->materialIds[i];

			std::string materialName = "None (Material)";
			if (materialIds[i] != PhysicsEngine::Guid::INVALID) {
				materialName = materialIds[i].toString();
			}

			bool materialSlotFillable = clipboard.getDraggedType() == InteractionType::Material;
			bool materialSlotFilled = false;
			bool materialIsClicked = ImGui::Slot("Material", materialName, materialSlotFillable, &materialSlotFilled);

			if (materialSlotFilled) {
				materialIds[i] = clipboard.getDraggedId();
				clipboard.clearDraggedItem();
			}

			CommandManager::addCommand(new ChangePropertyCommand<Guid>(&meshRenderer->materialIds[i], materialIds[i]));
		}

		ImGui::TreePop();
	}
}