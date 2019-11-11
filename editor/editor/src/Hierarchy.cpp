#include "../include/Hierarchy.h"
#include "../include/EditorCommands.h"
#include "../include/CommandManager.h"

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

void Hierarchy::render(World* world, const EditorScene scene, bool isOpenedThisFrame)
{
	static bool hierarchyActive = true;

	if (isOpenedThisFrame){
		hierarchyActive = true;

		entities.clear();
	}

	int numberOfEntities = world->getNumberOfEntities();
	entities.resize(numberOfEntities);
	for (int i = 0; i < numberOfEntities; i++) {
		entities[i] = *world->getEntityByIndex(i);
	}

	if (!hierarchyActive){
		return;
	}

	if (ImGui::Begin("Hierarchy", &hierarchyActive))
	{
		if (scene.name.length() > 0) {
			if (scene.isDirty) {
				ImGui::Text((scene.name+"*").c_str());
			}
			else {
				ImGui::Text(scene.name.c_str());
			}
			ImGui::Separator();

			if (entities.size() == 1) {
				selectedEntity = NULL;
			}

			// skip editor camera entity
			for (size_t i = 1; i < entities.size(); i++) {
				std::string name = entities[i].entityId.toString();

				static bool selected = false;
				if (ImGui::Selectable(name.c_str(), &selected)) {
					selectedEntity = &entities[i];
				}
			}

			if (ImGui::BeginPopupContextWindow("RightMouseClickPopup")) {
				if (ImGui::MenuItem("Copy", NULL, false, selectedEntity != NULL)) {
					
				}
				if (ImGui::MenuItem("Paste", NULL, false, selectedEntity != NULL))
				{
					
				}
				if (ImGui::MenuItem("Delete", NULL, false, selectedEntity != NULL) && selectedEntity != NULL)
				{
					world->latentDestroyEntity(selectedEntity->entityId);
				}

				ImGui::Separator();

				if (ImGui::BeginMenu("Create..."))
				{
					if (ImGui::MenuItem("Empty")) {
						CommandManager::addCommand(new CreateEntityCommand(world));
					}
					if (ImGui::MenuItem("Camera")) {
						CommandManager::addCommand(new CreateCameraCommand(world));
					}
					if (ImGui::MenuItem("Light")) {
						CommandManager::addCommand(new CreateLightCommand(world));
					}

					if (ImGui::BeginMenu("3D")) {
						if (ImGui::MenuItem("Cube")) {
							CommandManager::addCommand(new CreateCubeCommand(world));
						}
						if (ImGui::MenuItem("Sphere")) {
							CommandManager::addCommand(new CreateSphereCommand(world));
						}
						ImGui::EndMenu();
					}
					
					ImGui::EndMenu();
				}

				ImGui::EndPopup();
			}
		}
	}

	ImGui::End();
}

Entity* Hierarchy::getSelectedEntity()
{
	return selectedEntity;
}