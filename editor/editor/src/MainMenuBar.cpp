#include "../include/MainMenuBar.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

MainMenuBar::MainMenuBar()
{
	newClicked = false;
	openClicked = false;
	saveClicked = false;
	saveAsClicked = false;
	quitClicked = false;
	openInspectorClicked = false;
	openHierarchyClicked = false;
	aboutClicked = false;
}

MainMenuBar::~MainMenuBar()
{

}

void MainMenuBar::render()
{
	newClicked = false;
	openClicked = false;
	saveClicked = false;
	saveAsClicked = false;
	quitClicked = false;
	openInspectorClicked = false;
	openHierarchyClicked = false;
	aboutClicked = false;

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			showMenuFile();
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("Edit"))
		{
			showMenuEdit();
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("Windows")){
			showMenuWindow();
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("Help")){
			showMenuHelp();
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	if (openClicked){
		filebrowser.setMode(FilebrowserMode::Open);
	}
	else if (saveAsClicked){
		filebrowser.setMode(FilebrowserMode::Save);
	}

	filebrowser.render(openClicked | saveAsClicked);

	if (filebrowser.isOpenClicked()) {

	}
	else if (filebrowser.isSaveClicked()) {

	}

	aboutPopup.render(aboutClicked);
}

bool MainMenuBar::isNewClicked()
{
	return newClicked;
}

bool MainMenuBar::isOpenClicked()
{
	return openClicked;
}

bool MainMenuBar::isSaveClicked()
{
	return saveClicked;
}

bool MainMenuBar::isSaveAsClicked()
{
	return saveAsClicked;
}

bool MainMenuBar::isQuitClicked()
{
	return quitClicked;
}

bool MainMenuBar::isFilebrowserOpenClicked()
{
	return filebrowser.isOpenClicked();
}

bool MainMenuBar::isFilebrowserSaveClicked()
{
	return filebrowser.isSaveClicked();
}

bool MainMenuBar::isOpenInspectorCalled()
{
	return openInspectorClicked;
}

bool MainMenuBar::isOpenHierarchyCalled()
{
	return openHierarchyClicked;
}

bool MainMenuBar::isAboutClicked()
{
	return aboutClicked;
}

std::string MainMenuBar::getOpenFile()
{
	return filebrowser.getOpenFile();
}

std::string MainMenuBar::getSaveFile()
{
	return filebrowser.getSaveFile();
}

void MainMenuBar::showMenuFile()
{
	ImGui::MenuItem("(dummy menu)", NULL, false, false);
	if (ImGui::MenuItem("New")) {
		newClicked = true;
	}
	if (ImGui::MenuItem("Open", "Ctrl+O"))
	{
		openClicked = true;
	}
	if (ImGui::MenuItem("Save", "Ctrl+S")) {
		saveClicked = true;
	}
	if (ImGui::MenuItem("Save As..")) {
		saveAsClicked = true;
	}

	ImGui::Separator();
	if (ImGui::BeginMenu("Options"))
	{
		static bool enabled = true;
		ImGui::MenuItem("Enabled", "", &enabled);
		ImGui::BeginChild("child", ImVec2(0, 60), true);
		for (int i = 0; i < 10; i++)
			ImGui::Text("Scrolling Text %d", i);
		ImGui::EndChild();
		static float f = 0.5f;
		static int n = 0;
		static bool b = true;
		ImGui::SliderFloat("Value", &f, 0.0f, 1.0f);
		ImGui::InputFloat("Input", &f, 0.1f);
		ImGui::Combo("Combo", &n, "Yes\0No\0Maybe\0\0");
		ImGui::Checkbox("Check", &b);
		ImGui::EndMenu();
	}
	if (ImGui::BeginMenu("Colors"))
	{
		float sz = ImGui::GetTextLineHeight();
		for (int i = 0; i < ImGuiCol_COUNT; i++)
		{
			const char* name = ImGui::GetStyleColorName((ImGuiCol)i);
			ImVec2 p = ImGui::GetCursorScreenPos();
			ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + sz, p.y + sz), ImGui::GetColorU32((ImGuiCol)i));
			ImGui::Dummy(ImVec2(sz, sz));
			ImGui::SameLine();
			ImGui::MenuItem(name);
		}
		ImGui::EndMenu();
	}

	if (ImGui::MenuItem("Quit", "Alt+F4")) {
		quitClicked = true;
	}
}

void MainMenuBar::showMenuEdit()
{
	if (ImGui::MenuItem("Undo", "CTRL+Z")) {}
	if (ImGui::MenuItem("Redo", "CTRL+Y", false, false)) {}  // Disabled item
	ImGui::Separator();
	if (ImGui::MenuItem("Cut", "CTRL+X")) {}
	if (ImGui::MenuItem("Copy", "CTRL+C")) {}
	if (ImGui::MenuItem("Paste", "CTRL+V")) {}
}

void MainMenuBar::showMenuWindow()
{
	if (ImGui::MenuItem("Heirarchy")){
		openHierarchyClicked = true;
	}
	if (ImGui::MenuItem("Inspector")){
		openInspectorClicked = true;
	}
}

void MainMenuBar::showMenuHelp()
{
	if (ImGui::MenuItem("About PhysicsEngine")) {
		aboutClicked = true;
	}
}