#include "../include/Inspector.h"
#include "../include/LoadInspectorDrawerInternal.h"
#include "../include/FileSystemUtil.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"

#include "../include/components/MeshRenderer.h"
#include "../include/components/Light.h"

using namespace PhysicsEditor;

Inspector::Inspector()
{

}

Inspector::~Inspector()
{

}

void Inspector::render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, bool isOpenedThisFrame)
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
		// draw entity
		if (clipboard.getSelectedType() == InteractionType::Entity) {
			drawEntity(world, project, scene, clipboard);
		}
		
		// draw asset
		if(clipboard.getSelectedType() == InteractionType::Material){
			materialDrawer.render(world, project, scene, clipboard, clipboard.getSelectedId());
		}
		else if(clipboard.getSelectedType() == InteractionType::Shader){
			shaderDrawer.render(world, project, scene, clipboard, clipboard.getSelectedId());
		}
		else if (clipboard.getSelectedType() == InteractionType::Texture2D) {
			texture2DDrawer.render(world, project, scene, clipboard, clipboard.getSelectedId());
		}

		ImGui::Separator();
	}

	ImGui::End();
}

void Inspector::drawEntity(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard)
{
	Entity* entity = world->getEntityById(clipboard.getSelectedId());

	// entity may have been recently deleted
	if (entity == NULL) {
		return;
	}

	std::vector<std::pair<Guid, int>> componentsOnEntity = entity->getComponentsOnEntity(world);
	for (size_t i = 0; i < componentsOnEntity.size(); i++)
	{
		Guid componentId = componentsOnEntity[i].first;
		int componentType = componentsOnEntity[i].second;

		InspectorDrawer* drawer = NULL;
		if (Component::isInternal(componentType)) {
			drawer = loadInternalInspectorComponentDrawer(componentType);
		}
		else {
			//drawer = loadInspectorDrawer(componentType);
		}

		drawer->render(world, project, scene, clipboard, componentId);
		ImGui::Separator();

		delete drawer;
	}

	std::string componentToAdd = "";
	if (BeginAddComponentDropdown("Add component", componentToAdd)) {

		if (componentToAdd == "Transform") {
			CommandManager::addCommand(new AddComponentCommand<Transform>(world, entity->getId(), &scene.isDirty));
		}
		else if (componentToAdd == "Rigidbody") {
			CommandManager::addCommand(new AddComponentCommand<Rigidbody>(world, entity->getId(), &scene.isDirty));
		}
		else if (componentToAdd == "Camera") {
			CommandManager::addCommand(new AddComponentCommand<Camera>(world, entity->getId(), &scene.isDirty));
		}
		else if (componentToAdd == "MeshRenderer") {
			CommandManager::addCommand(new AddComponentCommand<MeshRenderer>(world, entity->getId(), &scene.isDirty));
		}
		else if (componentToAdd == "Light") {
			CommandManager::addCommand(new AddComponentCommand<Light>(world, entity->getId(), &scene.isDirty));
		}

		EndAddComponentDropdown();
	}
}

// void drawAsset(World* world, EditorScene& scene, EditorClipboard& clipboard)
// {
// 	Texture2D* texture = world->getAsset<Texture2D>(clipboard.getSelectedId());

// 	InspectorDrawer* drawer = loadInternalInspectorAssetDrawer(AssetType<Texture2D>::type);

// 	//drawer->render(world, clipboard, entity->entityId, componentId);
// 	ImGui::Separator();

// 	delete drawer;
// }

void drawCodeFile(World* world, EditorScene& scene, EditorClipboard& clipboard)
{

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