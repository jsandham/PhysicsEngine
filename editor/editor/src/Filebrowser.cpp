#define NOMINMAX

#include <algorithm>

#include "../include/FileBrowser.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

Filebrowser::Filebrowser()
{
	wasVisible = false;

	currentPath = currentWorkingDirectory();

	inputBuffer.resize(256);

	currentFilter = ".";
}

Filebrowser::~Filebrowser()
{

}

void Filebrowser::render(bool isVisible)
{
	if (wasVisible != isVisible){
		wasVisible = isVisible;
		if (isVisible){
			currentFiles = getFilesInDirectory(currentPath);

			ImGui::SetNextWindowSizeConstraints(ImVec2(500.0f, 400.0f), ImVec2(1920.0f, 1080.0f));
			ImGui::OpenPopup("Filebrowser");
		}
	}

	if (ImGui::BeginPopupModal("Filebrowser"))
	{
		float windowWidth = ImGui::GetWindowWidth();

		ImGuiTextFilter textFilter(currentFilter.c_str());
		std::vector<std::string> filteredCurrentFiles;
		for (size_t i = 0; i < currentFiles.size(); i++){
			if (textFilter.PassFilter(currentFiles[i].c_str()))
			{
				filteredCurrentFiles.push_back(currentFiles[i]);
			}
		}

		if (filteredCurrentFiles.size() == 0){
			filteredCurrentFiles.push_back("");
		}

		std::vector<const char*> cStrFilteredCurrentFiles;
		for (size_t i = 0; i < filteredCurrentFiles.size(); ++i)
		{
			cStrFilteredCurrentFiles.push_back(filteredCurrentFiles[i].c_str());
		}

		ImGui::Text(std::to_string(windowWidth).c_str());

		ImGui::PushItemWidth(windowWidth);
		static int selection = 0;
		if (ImGui::ListBox("##Current directory", &selection, &cStrFilteredCurrentFiles[0], (int)cStrFilteredCurrentFiles.size(), 10)) {
			for (int i = 0; i < std::min(256, (int)filteredCurrentFiles[selection].length()); i++){
				inputBuffer[i] = filteredCurrentFiles[selection][i];
			}
			for (int i = std::min(256, (int)filteredCurrentFiles[selection].length()); i < 256; i++){
				inputBuffer[i] = ' ';
			}
		}
		ImGui::PopItemWidth();

		float fileNameTitleWidth = 80.0f;
		float filterDropDownWidth = 200.0f;
		float inputTextWidth = windowWidth - fileNameTitleWidth - filterDropDownWidth - 10.0f;

		ImGui::SetNextItemWidth(fileNameTitleWidth);
		ImGui::Text("File Name");
		ImGui::SameLine();
		ImGui::SetNextItemWidth(inputTextWidth);
		if (ImGui::InputText("##File Name", &inputBuffer[0], (int)inputBuffer.size(), ImGuiInputTextFlags_EnterReturnsTrue)){

		}
		ImGui::SameLine();
		const char* filterNames[] = { "Text Files (.txt)", 
									  "Obj Files (.obj)", 
									  "Scene Files (.scene)", 
									  "JSON Files (.json)", 
									  "All Files (*)",
									  "IniFiles (.ini)"};
		const char* filters[] = { ".txt",
								  ".obj",
								  ".scene",
								  ".json",
								  ".",
								  ".ini"};
		static int filterIndex = 4;
		ImGui::SetNextItemWidth(filterDropDownWidth);
		ImGui::Combo("##Filter", &filterIndex, filterNames, IM_ARRAYSIZE(filterNames));
		currentFilter = filters[filterIndex];

		if (ImGui::Button("Open")){

		}
		ImGui::SameLine();
		if (ImGui::Button("Cancel")){
			ImGui::CloseCurrentPopup();
		}

		ImGui::EndPopup();
	}
}










bool Filebrowser::BeginFilterDropdown(std::string filter)
{
	ImGui::SameLine(0.f, 0.f);
	ImGui::PushID("##Filter");
	//bool pressed = ImGui::Button(&filter[0], ImVec2(100, 19));
	bool pressed = ImGui::Button(&filter[0]);
	ImGui::PopID();

	if (pressed)
	{
		ImGui::OpenPopup("##Filter");
	}

	if (ImGui::BeginPopup("##Filter"))
	{
		const char* test[] = { "A", "B", "C", "All Files (*)" };
		int s = 0;
		if (ImGui::ListBox("##Filter", &s, test, IM_ARRAYSIZE(test), 4)) {
			currentFilter = test[s];
			ImGui::CloseCurrentPopup();
		}
		return true;
	}

	return false;
}

void Filebrowser::EndFilterDropdown()
{
	ImGui::EndPopup();
}








//bool Filebrowser::BeginButtonDropDown(const char* label, ImVec2 buttonSize)
//{
//	ImGui::SameLine(0.f, 0.f);
//
//	ImGuiWindow* window = GetCurrentWindow();
//	ImGuiState& g = *GImGui;
//	const ImGuiStyle& style = g.Style;
//
//	float x = ImGui::GetCursorPosX();
//	float y = ImGui::GetCursorPosY();
//
//	ImVec2 size(20, buttonSize.y);
//	bool pressed = ImGui::Button("##", size);
//
//	// Arrow
//	ImVec2 center(window->Pos.x + x + 10, window->Pos.y + y + buttonSize.y / 2);
//	float r = 8.f;
//	center.y -= r * 0.25f;
//	ImVec2 a = center + ImVec2(0, 1) * r;
//	ImVec2 b = center + ImVec2(-0.866f, -0.5f) * r;
//	ImVec2 c = center + ImVec2(0.866f, -0.5f) * r;
//
//	window->DrawList->AddTriangleFilled(a, b, c, GetColorU32(ImGuiCol_Text));
//
//	// Popup
//
//	ImVec2 popupPos;
//
//	popupPos.x = window->Pos.x + x - buttonSize.x;
//	popupPos.y = window->Pos.y + y + buttonSize.y;
//
//	ImGui::SetNextWindowPos(popupPos);
//
//	if (pressed)
//	{
//		ImGui::OpenPopup(label);
//	}
//
//	if (ImGui::BeginPopup(label))
//	{
//		ImGui::PushStyleColor(ImGuiCol_FrameBg, style.Colors[ImGuiCol_Button]);
//		ImGui::PushStyleColor(ImGuiCol_WindowBg, style.Colors[ImGuiCol_Button]);
//		ImGui::PushStyleColor(ImGuiCol_ChildWindowBg, style.Colors[ImGuiCol_Button]);
//		return true;
//	}
//
//	return false;
//}
//
//void Filebrowser::EndButtonDropDown()
//{
//	ImGui::PopStyleColor(3);
//	ImGui::EndPopup();
//}


/*ImGui::TextWrapped("Enter 'HELP' for help, press TAB to use text completion.");*/
/*ImGui::TextWrapped(currentFiles[0].c_str());*/
//ImGui::TextWrapped(std::to_string(currentFiles.size()).c_str());