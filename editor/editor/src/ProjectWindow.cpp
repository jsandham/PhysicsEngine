#include "../include/ProjectWindow.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

ProjectWindow::ProjectWindow()
{
	isVisible = false;
	openClicked = false;
	createClicked = false;
	mode = ProjectWindowMode::OpenProject;

	inputBuffer.resize(256);

	filebrowser.setMode(FilebrowserMode::SelectFolder);
}

ProjectWindow::~ProjectWindow()
{

}

void ProjectWindow::render(bool becomeVisibleThisFrame)
{
	openClicked = false;
	createClicked = false;

	if (isVisible != becomeVisibleThisFrame) {
		isVisible = becomeVisibleThisFrame;
		if (becomeVisibleThisFrame) {
			//ImGui::SetNextWindowSize(ImVec2(1000.0f, 1000.0f));
			ImGui::SetNextWindowSizeConstraints(ImVec2(500.0f, 200.0f), ImVec2(1920.0f, 1080.0f));
			//ImGui::SetNextWindowBgAlpha(1.0f);

			ImGui::OpenPopup("##ProjectWindow");
		}
	}

	bool projectWindowOpen = true;
	if (ImGui::BeginPopupModal("##ProjectWindow", &projectWindowOpen))
	{
		float windowWidth = ImGui::GetWindowWidth();

		if (mode == ProjectWindowMode::NewProject) {
			renderNewMode();
		}
		else if (mode == ProjectWindowMode::OpenProject) {
			renderOpenMode();
		}

		ImGui::EndPopup();
	}
}

bool ProjectWindow::isOpenClicked() const
{
	return openClicked;
}

bool ProjectWindow::isCreateClicked() const
{
	return createClicked;
}

std::string ProjectWindow::getProjectName() const
{
	int index = 0;
	for (size_t i = 0; i < inputBuffer.size(); i++) {
		if (inputBuffer[i] == '\0') {
			index = (int)i;
			break;
		}
	}
	return std::string(inputBuffer.begin(), inputBuffer.begin() + index);
}

std::string ProjectWindow::getSelectedFolderPath() const
{
	return filebrowser.getSelectedFolderPath();
}

void ProjectWindow::renderNewMode()
{
	float projectNameTitleWidth = 100.0f;
	float inputTextWidth = 400.0f;

	ImGui::SetNextItemWidth(projectNameTitleWidth);
	ImGui::Text("Project Name");
	ImGui::SameLine();
	ImGui::SetNextItemWidth(inputTextWidth);
	if (ImGui::InputText("##Project Name", &inputBuffer[0], (int)inputBuffer.size(), ImGuiInputTextFlags_EnterReturnsTrue)) {

	}

	bool openSelectFolderBrowser = false;
	if (ImGui::Button("Select Folder")) {
		openSelectFolderBrowser = true;
	}

	ImGui::SameLine();
	ImGui::Text(filebrowser.getSelectedFolderPath().c_str());

	filebrowser.render(openSelectFolderBrowser);

	if (ImGui::Button("Create Project")) {
		createClicked = true;
		ImGui::CloseCurrentPopup();
	}
}

void ProjectWindow::renderOpenMode()
{
	bool openSelectFolderBrowser = false;
	if (ImGui::Button("Select Folder")) {
		openSelectFolderBrowser = true;
	}

	ImGui::SameLine();
	ImGui::Text(filebrowser.getSelectedFolderPath().c_str());

	filebrowser.render(openSelectFolderBrowser);

	if (ImGui::Button("Open Project")) {
		openClicked = true;
		ImGui::CloseCurrentPopup();
	}
}

void ProjectWindow::setMode(ProjectWindowMode mode)
{
	this->mode = mode;
}