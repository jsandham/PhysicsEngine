#include "../include/Hierarchy.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

Hierarchy::Hierarchy()
{
	selectedEntity = NULL;
}

Hierarchy::~Hierarchy()
{

}

void Hierarchy::render(World world, bool isOpenedThisFrame)
{
	static bool hierarchyActive = true;

	if (isOpenedThisFrame){
		hierarchyActive = true;

		entities.clear();

		int numberOfEntities = world.getNumberOfEntities();

		entities.resize(numberOfEntities);
		for (int i = 0; i < numberOfEntities; i++) {
			entities[i] = *world.getEntityByIndex(i);
		}
	}

	if (!hierarchyActive){
		return;
	}

	if (ImGui::Begin("Hierarchy", &hierarchyActive))
	{
		for (size_t i = 0; i < entities.size(); i++) {
			std::string name = entities[i].entityId.toString();
			
			static bool selected = false;
			if (ImGui::Selectable(name.c_str(), &selected)) {
			}
		}

		if (ImGui::IsWindowHovered()) {
			ImGui::Text("hierarchy hovered");

			ImGuiIO& io = ImGui::GetIO();

			if (ImGui::IsMouseClicked(1))
			{
				ImGui::Text("Mouse clicked:");

				ImGui::OpenPopup("HierarchyPopupWindow");
				ImGui::SameLine(); ImGui::Text("b%d", 1);
			}

			/*if (ImGui::BeginPopupModal("HierarchyPopupWindow")) {

				ImGui::EndPopup();
			}*/
		}

		ImGui::End();
	}
}

Entity* Hierarchy::getSelectedEntity()
{
	return selectedEntity;
}