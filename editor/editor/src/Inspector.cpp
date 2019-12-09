#include "../include/Inspector.h"
#include "../include/LoadInspectorDrawerInternal.h"
#include "../include/FileSystemUtil.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

Inspector::Inspector()
{
	
}

Inspector::~Inspector()
{
	
}

void Inspector::render(World* world, Entity* entity, EditorScene& scene, EditorUI& ui, bool isOpenedThisFrame)
{
	static bool inspectorActive = true;

	if (isOpenedThisFrame){
		inspectorActive = true;
	}

	if (!inspectorActive){
		return;
	}

	if (ImGui::Begin("Inspector", &inspectorActive))
	{
		if (entity != NULL){
			std::vector<std::pair<Guid, int>> componentsOnEntity = entity->getComponentsOnEntity(world);
			for (size_t i = 0; i < componentsOnEntity.size(); i++)
			{
				Guid componentId = componentsOnEntity[i].first;
				int componentType = componentsOnEntity[i].second;

				InspectorDrawer* drawer = NULL;
				if (componentType < 20) {
					drawer = loadInternalInspectorDrawer(componentType);
				}
				else {
					//drawer = loadInspectorDrawer(componentType);
				}

				drawer->render(world, ui, entity->entityId, componentId);
				ImGui::Separator();

				delete drawer;
			}

			std::string componentToAdd = "";
			if (BeginAddComponentDropdown("Add component", componentToAdd)){

				if (componentToAdd == "Transform") {
					scene.isDirty = true; // actually should I pass this through to be modified in the command?
					CommandManager::addCommand(new AddComponentCommand<Transform>(world, entity->entityId));
				}
				else if (componentToAdd == "Rigidbody") {
					scene.isDirty = true;
					CommandManager::addCommand(new AddComponentCommand<Rigidbody>(world, entity->entityId));
				}
				else if (componentToAdd == "Camera") {
					scene.isDirty = true;
					CommandManager::addCommand(new AddComponentCommand<Camera>(world, entity->entityId));
				}
				else if (componentToAdd == "MeshRenderer") {
					scene.isDirty = true;
					CommandManager::addCommand(new AddComponentCommand<MeshRenderer>(world, entity->entityId));
				}
				else if (componentToAdd == "Light") {
					scene.isDirty = true;
					CommandManager::addCommand(new AddComponentCommand<Light>(world, entity->entityId));
				}
				
				EndAddComponentDropdown();
			}
		}
	}

	ImGui::End();
}

bool Inspector::BeginAddComponentDropdown(std::string name, std::string& componentToAdd)
{
	ImGui::PushID("##Dropdown");
	bool pressed = ImGui::Button(name.c_str());
	ImGui::PopID();

	if (pressed)
	{
		ImGui::OpenPopup("##Dropdown");
	}

	if (ImGui::BeginPopup("##Dropdown"))
	{
		std::vector<char> inputBuffer(128);
		if (ImGui::InputTextWithHint("##Search string", "search...", &inputBuffer[0], (int)inputBuffer.size())){
			
		}

		std::vector<std::string> components = { "Transform",
												"Camera",
												"Light",
												"Rigidbody",
												"MeshRenderer",
												"LineRenderer",
												"BoxCollider",
												"SphereCollider"};

		ImGuiTextFilter componentFilter(&inputBuffer[0]);
		std::vector<std::string> filteredComponents;
		for (size_t i = 0; i < components.size(); i++){
			if (componentFilter.PassFilter(components[i].c_str()))
			{
				filteredComponents.push_back(components[i]);
			}
		}

		if (filteredComponents.size() == 0){
			filteredComponents.push_back("");
		}

		std::vector<const char*> cStrFilteredComponents;
		for (size_t i = 0; i < filteredComponents.size(); ++i)
		{
			cStrFilteredComponents.push_back(filteredComponents[i].c_str());
		}

		int s = 0;
		if (ImGui::ListBox("##Filter", &s, &cStrFilteredComponents[0], (int)cStrFilteredComponents.size(), 4)) {
			componentToAdd = filteredComponents[s];
			ImGui::CloseCurrentPopup();
		}
		return true;
	}

	return false;
}

void Inspector::EndAddComponentDropdown()
{
	ImGui::EndPopup();
}