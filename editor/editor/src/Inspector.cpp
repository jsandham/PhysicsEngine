#include "../include/Inspector.h"
#include "../include/LoadInspectorDrawerInternal.h"

#include "../include/CameraDrawer.h"

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

void Inspector::render(Entity* entity, bool isOpenedThisFrame)
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
		// make static?
		InspectorDrawer* transformDrawer = loadInternalInspectorDrawer(0);
		InspectorDrawer* cameraDrawer = loadInternalInspectorDrawer(2);
		InspectorDrawer* lightDrawer = loadInternalInspectorDrawer(5);

		transformDrawer->render(&transform);
		cameraDrawer->render(&camera);
		lightDrawer->render(&light);

		delete cameraDrawer;
		delete lightDrawer;


		if (entity != NULL){
			std::string componentToAdd = "";
			if (BeginAddComponentDropdown("Add component", componentToAdd)){


				EndAddComponentDropdown();
			}
		}

		ImGui::End();
	}
}

bool Inspector::BeginAddComponentDropdown(std::string name, std::string& componentToAdd)
{
	ImGui::SameLine(0.f, 0.f);
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