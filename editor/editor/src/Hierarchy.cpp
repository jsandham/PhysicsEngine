#include "../include/Hierarchy.h"
#include "../include/EditorCommands.h"
#include "../include/CommandManager.h"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

Hierarchy::Hierarchy()
{

}

Hierarchy::~Hierarchy()
{

}

void Hierarchy::render(World* world, EditorScene& scene, EditorClipboard& clipboard, bool isOpenedThisFrame)
{
	static bool hierarchyActive = true;

	if (isOpenedThisFrame){
		hierarchyActive = true;

		entities.clear();
	}

	int numberOfEntities = world->getNumberOfEntities();
	if (entities.size() != numberOfEntities) {
		entities.resize(numberOfEntities);
		entityNames.resize(numberOfEntities);
		for (int i = 0; i < numberOfEntities; i++) {
			Entity* entity = world->getEntityByIndex(i);

			entities[i] = *entity;
			entityNames[i] = entity->getId().toString();
		}
	}
	/*int numberOfEntities = world->getNumberOfEntities();
	entities.resize(numberOfEntities);
	for (int i = 0; i < numberOfEntities; i++) {
		entities[i] = *world->getEntityByIndex(i);
	}*/

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

			// skip editor camera entity
			for (size_t i = 1; i < entities.size(); i++) {
				//std::string name = entities[i].entityId.toString();

				static bool selected = false;
				if (ImGui::Selectable(entityNames[i].c_str(), &selected)) {
					clipboard.setSelectedItem(InteractionType::Entity, entities[i].getId());
				}
			}

			if (ImGui::BeginPopupContextWindow("RightMouseClickPopup")) {
				if (ImGui::MenuItem("Copy", NULL, false, clipboard.getSelectedType() == InteractionType::Entity)) {
					
				}
				if (ImGui::MenuItem("Paste", NULL, false, clipboard.getSelectedType() == InteractionType::Entity))
				{
					
				}
				if (ImGui::MenuItem("Delete", NULL, false, clipboard.getSelectedType() == InteractionType::Entity) && clipboard.getSelectedType() == InteractionType::Entity)
				{
					world->latentDestroyEntity(clipboard.getSelectedId());
				}

				ImGui::Separator();

				if (ImGui::BeginMenu("Create..."))
				{
					if (ImGui::MenuItem("Empty")) {
						CommandManager::addCommand(new CreateEntityCommand(world, &scene.isDirty));
					}
					if (ImGui::MenuItem("Camera")) {
						CommandManager::addCommand(new CreateCameraCommand(world, &scene.isDirty));
					}
					if (ImGui::MenuItem("Light")) {
						CommandManager::addCommand(new CreateLightCommand(world, &scene.isDirty));
					}

					if (ImGui::BeginMenu("3D")) {
						if (ImGui::MenuItem("Cube")) {
							CommandManager::addCommand(new CreateCubeCommand(world, &scene.isDirty));
						}
						if (ImGui::MenuItem("Sphere")) {
							CommandManager::addCommand(new CreateSphereCommand(world, &scene.isDirty));
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