#include <algorithm>

#include "../include/imgui_extensions.h"

using namespace ImGui;

void ImGui::ToggleButton(const char* str_id, bool* v)
{
	ImVec2 p = ImGui::GetCursorScreenPos();
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	float height = ImGui::GetFrameHeight();
	float width = height * 1.55f;
	float radius = height * 0.50f;

	ImGui::InvisibleButton(str_id, ImVec2(width, height));
	if (ImGui::IsItemClicked())
		*v = !*v;

	float t = *v ? 1.0f : 0.0f;

	ImGuiContext& g = *GImGui;
	float ANIM_SPEED = 0.08f;
	if (g.LastActiveId == g.CurrentWindow->GetID(str_id))// && g.LastActiveIdTimer < ANIM_SPEED)
	{
		float t_anim = ImSaturate(g.LastActiveIdTimer / ANIM_SPEED);
		t = *v ? (t_anim) : (1.0f - t_anim);
	}

	ImU32 col_bg;
	if (ImGui::IsItemHovered())
		col_bg = ImGui::GetColorU32(ImLerp(ImVec4(0.78f, 0.78f, 0.78f, 1.0f), ImVec4(0.64f, 0.83f, 0.34f, 1.0f), t));
	else
		col_bg = ImGui::GetColorU32(ImLerp(ImVec4(0.85f, 0.85f, 0.85f, 1.0f), ImVec4(0.56f, 0.83f, 0.26f, 1.0f), t));

	draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5f);
	draw_list->AddCircleFilled(ImVec2(p.x + radius + t * (width - radius * 2.0f), p.y + radius), radius - 1.5f, IM_COL32(255, 255, 255, 255));
}















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


static auto vector_getter = [](void* vec, int idx, const char** out_text)
{
	auto& vector = *static_cast<std::vector<std::string>*>(vec);
	if (idx < 0 || idx >= static_cast<int>(vector.size())) { return false; }
	*out_text = vector.at(idx).c_str();
	return true;
};

bool ImGui::Combo(const char* label, int* currIndex, std::vector<std::string>& values)
{
	if (values.empty()) { return false; }
	return Combo(label, currIndex, vector_getter,
		static_cast<void*>(&values), (int)values.size());
}



bool ImGui::Slot(const std::string slotLabel, const std::string slotText, bool slotFillable, bool* slotFilled)
{
	ImVec2 windowSize = ImGui::GetWindowSize();
	windowSize.x = std::min(std::max(windowSize.x - 100.0f, 50.0f), 250.0f);

	ImGui::ButtonEx(slotText.c_str(), ImVec2(windowSize.x, 0), ImGuiButtonFlags_Disabled);
	//ImGui::ButtonEx(slotText.c_str(), windowSize, ImGuiButtonFlags_Disabled);
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

bool ImGui::ImageSlot(const std::string slotLabel, GLuint texture, bool slotFillable, bool* slotFilled)
{
	ImGui::ImageButton((void*)(intptr_t)texture, ImVec2(80, 80), ImVec2(1, 1), ImVec2(0, 0), 0, ImVec4(1, 1, 1, 1), ImVec4(1, 1, 1, 0.5));

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