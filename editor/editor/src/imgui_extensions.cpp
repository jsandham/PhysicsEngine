#include "../include/imgui_extensions.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"


bool BeginDropdown(std::string name, std::vector<std::string> values, int* selection)
{
	ImGui::SameLine(0.f, 0.f);
	ImGui::PushID(("##" + name).c_str());

	bool pressed = ImGui::Button(&name[0]);
	ImGui::PopID();

	if (pressed)
	{
		ImGui::OpenPopup(("##" + name).c_str());
	}

	if (ImGui::BeginPopup(("##" + name).c_str()))
	{
		std::vector<const char*> temp(values.size());
		for (size_t i = 0; i < values.size(); i++) {
			temp[i] = values[i].c_str();
		}
		if (ImGui::ListBox(("##" + name).c_str(), selection, &temp[0], (int)temp.size(), 4)) {
			ImGui::CloseCurrentPopup();
		}
		return true;
	}

	return false;
}

void EndDropdown()
{
	ImGui::EndPopup();
}