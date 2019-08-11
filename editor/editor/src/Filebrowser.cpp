#define NOMINMAX

#include <algorithm>

#include "../include/FileBrowser.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

#include "../include/imgui_extensions.h"

using namespace PhysicsEditor;

Filebrowser::Filebrowser()
{
	isVisible = false;
	openClicked = false;
	saveClicked = false;
	mode = FilebrowserMode::Open;

	currentDirectory = currentWorkingDirectory();

	inputBuffer.resize(256);

	currentFilter = ".";

	openFile = "";
	saveFile = "";
}

Filebrowser::~Filebrowser()
{

}

void Filebrowser::render(bool becomeVisibleThisFrame)
{
	openClicked = false;
	saveClicked = false;

	openFile = "";
	saveFile = "";

	if (isVisible != becomeVisibleThisFrame){
		isVisible = becomeVisibleThisFrame;
		if (becomeVisibleThisFrame){
			updateCurrentDirectory(currentDirectory);

			ImGui::SetNextWindowSizeConstraints(ImVec2(500.0f, 400.0f), ImVec2(1920.0f, 1080.0f));
			ImGui::OpenPopup("Filebrowser");
		}
	}

	if (ImGui::BeginPopupModal("Filebrowser"))
	{
		float windowWidth = ImGui::GetWindowWidth();

		ImGui::Text(currentDirectory.c_str());

		for (auto x: currentDirectories) {
			ImGui::Text(x.c_str());
		}

		std::vector<std::string> directoryNamesInCurrentDirectoryPath = split(currentDirectory, '\\');
		std::vector<std::string> directoryPathsInCurrentDirectoryPath = getDirectoryPaths(currentDirectory);

		for (size_t i = 0; i < directoryNamesInCurrentDirectoryPath.size(); i++){

			std::string directory = directoryPathsInCurrentDirectoryPath[directoryPathsInCurrentDirectoryPath.size() - i - 1];

			if (ImGui::Button(directoryNamesInCurrentDirectoryPath[i].c_str()))
			{
				currentDirectory = directory;

				currentFiles = getFilesInDirectory(currentDirectory);
				currentDirectories = getDirectoriesInDirectory(currentDirectory);
				directoryNamesInCurrentDirectoryPath = split(currentDirectory, '\\');
				directoryPathsInCurrentDirectoryPath = getDirectoryPaths(currentDirectory);
			}

			std::vector<std::string> directories = getDirectoriesInDirectory(directory);
			std::vector<std::string> directoryNames;
			for (size_t j = 0; j < directories.size(); j++) {
				directoryNames.push_back(directories[j].substr(directories[j].find_last_of("/\\") + 1));
			}

			if (directories.size() > 0) {
				ImGui::SameLine(0, 0);
				
				int s = -1;
				if (BeginDropdown(">##" + std::to_string(i), directoryNames, &s)) {
					if (s >= 0) {
						currentDirectory = directories[s];

						currentFiles = getFilesInDirectory(currentDirectory);
						currentDirectories = getDirectoriesInDirectory(currentDirectory);
						directoryNamesInCurrentDirectoryPath = split(currentDirectory, '\\');
						directoryPathsInCurrentDirectoryPath = getDirectoryPaths(currentDirectory);
					}

					EndDropdown();
				}
			}

			if (i != directoryNamesInCurrentDirectoryPath.size() - 1){
				ImGui::SameLine(0, 0);
			}
		}
		
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

		ImGui::PushItemWidth(windowWidth);
		static int selection = 0;
		if (ImGui::ListBox("##Current directory contents", &selection, &cStrFilteredCurrentFiles[0], (int)cStrFilteredCurrentFiles.size(), 10)) {
			for (int i = 0; i < std::min(256, (int)filteredCurrentFiles[selection].length()); i++){
				inputBuffer[i] = filteredCurrentFiles[selection][i];
			}
			for (int i = std::min(256, (int)filteredCurrentFiles[selection].length()); i < 256; i++){
				inputBuffer[i] = ' ';
			}
		}
		ImGui::PopItemWidth();



		if (mode == FilebrowserMode::Open){
			renderOpenMode();
		}
		else if (mode == FilebrowserMode::Save){
			renderSaveMode();
		}

		ImGui::EndPopup();
	}
}

void Filebrowser::renderOpenMode()
{
	float windowWidth = ImGui::GetWindowWidth();

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
		"IniFiles (.ini)" };
	const char* filters[] = { ".txt",
		".obj",
		".scene",
		".json",
		".",
		".ini" };
	static int filterIndex = 4;
	ImGui::SetNextItemWidth(filterDropDownWidth);
	ImGui::Combo("##Filter", &filterIndex, filterNames, IM_ARRAYSIZE(filterNames));
	currentFilter = filters[filterIndex];

	if (ImGui::Button("Open")){
		openClicked = true;
		openFile = std::string(inputBuffer.begin(), inputBuffer.end());
	}
	ImGui::SameLine();
	if (ImGui::Button("Cancel")){
		ImGui::CloseCurrentPopup();
	}
}

void Filebrowser::renderSaveMode()
{
	float windowWidth = ImGui::GetWindowWidth();

	float fileNameTitleWidth = 120.0f;
	float saveAsTypeTitleWidth = 120.0f;
	float inputTextWidth = windowWidth - fileNameTitleWidth - 10.0f;
	float saveAsTypeDropDownWidth = windowWidth - fileNameTitleWidth - 10.0f;

	ImGui::SetNextItemWidth(fileNameTitleWidth);
	ImGui::Text("File Name");
	ImGui::SameLine();
	ImGui::SetNextItemWidth(inputTextWidth);
	if (ImGui::InputText("##File Name", &inputBuffer[0], (int)inputBuffer.size(), ImGuiInputTextFlags_EnterReturnsTrue)){

	}

	const char* saveAsTypeNames[] = { "Scene Files (.scene)",
									  "All Files (*)" };
	const char* saveAsTypes[] = { ".scene",
								  "."};

	static int saveAsTypeIndex = 4;
	
	ImGui::SetNextItemWidth(saveAsTypeTitleWidth);
	ImGui::Text("Save As Type");
	ImGui::SameLine();
	ImGui::SetNextItemWidth(saveAsTypeDropDownWidth);
	ImGui::Combo("##Filter", &saveAsTypeIndex, saveAsTypeNames, IM_ARRAYSIZE(saveAsTypeNames));

	//currentFilter = saveAsTypes[saveAsTypeIndex];

	if (ImGui::Button("Save")){
		saveClicked = true;
		saveFile = "";
	}
	ImGui::SameLine();
	if (ImGui::Button("Cancel")){
		ImGui::CloseCurrentPopup();
	}
}

void Filebrowser::setMode(FilebrowserMode mode)
{
	this->mode = mode;
}

std::string Filebrowser::getOpenFile()
{
	return openFile;
}

std::string Filebrowser::getSaveFile()
{
	return saveFile;
}

bool Filebrowser::isOpenClicked()
{
	return openClicked;
}

bool Filebrowser::isSaveClicked()
{
	return saveClicked;
}

void Filebrowser::updateCurrentDirectory(std::string currentDirectory)
{
	currentFiles = getFilesInDirectory(currentDirectory);
	currentDirectories = getDirectoriesInDirectory(currentDirectory);
	//currentDirectoryShortPaths = split(currentDirectory, '\\');
	//currentDirectoryLongPaths = getDirectoryLongPaths(currentDirectory);
}