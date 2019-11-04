#include "../include/MainMenuBar.h"
#include "../include/CommandManager.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

MainMenuBar::MainMenuBar()
{
	projectSelected = false;

	newSceneClicked = false;
	openSceneClicked = false;
	saveClicked = false;
	saveAsClicked = false;
	newProjectClicked = false;
	openProjectClicked = false;
	saveProjectClicked = false;
	buildClicked = false;
	quitClicked = false;
	openInspectorClicked = false;
	openHierarchyClicked = false;
	openConsoleClicked = false;
	openSceneViewClicked = false;
	openProjectViewClicked = false;
	aboutClicked = false;
}

MainMenuBar::~MainMenuBar()
{

}

void MainMenuBar::render(std::string currentProjectPath)
{
	projectSelected = currentProjectPath != "";

	newSceneClicked = false;
	openSceneClicked = false;
	saveClicked = false;
	saveAsClicked = false;
	newProjectClicked = false;
	openProjectClicked = false;
	saveProjectClicked = false;
	buildClicked = false;
	quitClicked = false;
	openInspectorClicked = false;
	openHierarchyClicked = false;
	openConsoleClicked = false;
	openSceneViewClicked = false;
	openProjectViewClicked = false;
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
}

bool MainMenuBar::isNewSceneClicked() const
{
	return newSceneClicked;
}

bool MainMenuBar::isOpenSceneClicked() const
{
	return openSceneClicked;
}

bool MainMenuBar::isSaveClicked() const
{
	return saveClicked;
}

bool MainMenuBar::isSaveAsClicked() const
{
	return saveAsClicked;
}

bool MainMenuBar::isBuildClicked() const
{
	return buildClicked;
}

bool MainMenuBar::isQuitClicked() const
{
	return quitClicked;
}

bool MainMenuBar::isNewProjectClicked() const
{
	return newProjectClicked;
}

bool MainMenuBar::isOpenProjectClicked() const
{
	return openProjectClicked;
}

bool MainMenuBar::isOpenInspectorCalled() const
{
	return openInspectorClicked;
}

bool MainMenuBar::isOpenHierarchyCalled() const
{
	return openHierarchyClicked;
}

bool MainMenuBar::isOpenConsoleCalled() const
{
	return openConsoleClicked;
}

bool MainMenuBar::isOpenSceneViewCalled() const
{
	return openSceneViewClicked;
}

bool MainMenuBar::isOpenProjectViewCalled() const
{
	return openProjectViewClicked;
}

bool MainMenuBar::isAboutClicked() const
{
	return aboutClicked;
}

void MainMenuBar::showMenuFile()
{
	if (ImGui::MenuItem("New Scene", NULL, false, projectSelected)) {
		newSceneClicked = true;
	}
	if (ImGui::MenuItem("Open Scene", "Ctrl+O", false, projectSelected))
	{
		openSceneClicked = true;
	}

	ImGui::Separator();

	if (ImGui::MenuItem("Save", "Ctrl+S", false, projectSelected)) {
		saveClicked = true;
	}
	if (ImGui::MenuItem("Save As..", NULL, false, projectSelected)) {
		saveAsClicked = true;
	}

	ImGui::Separator();

	if (ImGui::MenuItem("New Project")) {
		newProjectClicked = true;
	}
	if (ImGui::MenuItem("Open Project"))
	{
		openProjectClicked = true;
	}
	if (ImGui::MenuItem("Save Project", NULL, false, projectSelected)) {
		saveProjectClicked = true;
	}
	if (ImGui::MenuItem("Build", NULL, false, projectSelected)) {
		buildClicked = true;
	}

	if (ImGui::MenuItem("Quit", "Alt+F4")) {
		quitClicked = true;
	}
}

void MainMenuBar::showMenuEdit()
{
	if (ImGui::MenuItem("Undo", "CTRL+Z", false, CommandManager::canUndo())) {
		CommandManager::undoCommand();
	}
	if (ImGui::MenuItem("Redo", "CTRL+Y", false, CommandManager::canRedo())) {
		CommandManager::executeCommand();
	}
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
	if (ImGui::MenuItem("Console")){
		openConsoleClicked = true;
	}
	if (ImGui::MenuItem("Scene View")) {
		openSceneViewClicked = true;
	}
	if (ImGui::MenuItem("Project View")) {
		openProjectViewClicked = true;
	}
}

void MainMenuBar::showMenuHelp()
{
	if (ImGui::MenuItem("About PhysicsEngine")) {
		aboutClicked = true;
	}
}