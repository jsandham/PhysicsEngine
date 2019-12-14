#include <algorithm>

#include "../include/imgui_extensions.h"

using namespace ImGui;

bool ImGui::BeginDropdown(std::string name, std::vector<std::string> values, int* selection)
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

void ImGui::EndDropdown()
{
	ImGui::EndPopup();
}

bool ImGui::Slot(const std::string slotLabel, const std::string slotText, bool slotFillable, bool* slotFilled)
{
	ImVec2 windowSize = ImGui::GetWindowSize();
	windowSize.x = std::min(std::max(windowSize.x - 100.0f, 50.0f), 250.0f);

	ImGui::ButtonEx(slotText.c_str(), ImVec2(windowSize.x, 0), ImGuiButtonFlags_Disabled);
	ImVec2 size = ImGui::GetItemRectSize();
	ImVec2 position = ImGui::GetItemRectMin();

	ImVec2 topLeft = position;
	ImVec2 topRight = ImVec2(position.x + size.x, position.y);
	ImVec2 bottomLeft = ImVec2(position.x, position.y + size.y);
	ImVec2 bottomRight = ImVec2(position.x + size.x, position.y + size.y);

	ImGui::GetForegroundDrawList()->AddLine(topLeft, topRight, 0xFF0A0A0A);
	ImGui::GetForegroundDrawList()->AddLine(topRight, bottomRight, 0xFF333333);
	ImGui::GetForegroundDrawList()->AddLine(bottomRight, bottomLeft, 0xFF333333);
	ImGui::GetForegroundDrawList()->AddLine(bottomLeft, topLeft, 0xFF333333);

	size.x += position.x;
	size.y += position.y;

	bool isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly);
	bool isClicked = isHovered && ImGui::IsMouseClicked(0);

	if (isClicked) {
		ImGui::GetForegroundDrawList()->AddRect(position, size, 0xFFFF0000);
	}

	if (isHovered && slotFillable) {
		ImGui::GetForegroundDrawList()->AddRectFilled(position, size, 0x44FF0000);

		if (ImGui::IsMouseReleased(0)) {
			*slotFilled = true;
		}
	}

	ImGui::SameLine(); ImGui::Text(slotLabel.c_str());

	return isClicked;
}