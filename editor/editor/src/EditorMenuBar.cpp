#include "../include/EditorMenuBar.h"
#include "../include/CommandManager.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

EditorMenuBar::EditorMenuBar()
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
	preferencesClicked = false;
}

EditorMenuBar::~EditorMenuBar()
{

}

void EditorMenuBar::render(std::string currentProjectPath)
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
	preferencesClicked = false;

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

bool EditorMenuBar::isNewSceneClicked() const
{
	return newSceneClicked;
}

bool EditorMenuBar::isOpenSceneClicked() const
{
	return openSceneClicked;
}

bool EditorMenuBar::isSaveClicked() const
{
	return saveClicked;
}

bool EditorMenuBar::isSaveAsClicked() const
{
	return saveAsClicked;
}

bool EditorMenuBar::isBuildClicked() const
{
	return buildClicked;
}

bool EditorMenuBar::isQuitClicked() const
{
	return quitClicked;
}

bool EditorMenuBar::isNewProjectClicked() const
{
	return newProjectClicked;
}

bool EditorMenuBar::isOpenProjectClicked() const
{
	return openProjectClicked;
}

bool EditorMenuBar::isOpenInspectorCalled() const
{
	return openInspectorClicked;
}

bool EditorMenuBar::isOpenHierarchyCalled() const
{
	return openHierarchyClicked;
}

bool EditorMenuBar::isOpenConsoleCalled() const
{
	return openConsoleClicked;
}

bool EditorMenuBar::isOpenSceneViewCalled() const
{
	return openSceneViewClicked;
}

bool EditorMenuBar::isOpenProjectViewCalled() const
{
	return openProjectViewClicked;
}

bool EditorMenuBar::isAboutClicked() const
{
	return aboutClicked;
}

bool EditorMenuBar::isPreferencesClicked() const
{
	return preferencesClicked;
}

void EditorMenuBar::showMenuFile()
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

void EditorMenuBar::showMenuEdit()
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
	ImGui::Separator();
	if (ImGui::MenuItem("Preferences...")) {
		preferencesClicked = true;
	}
}

void EditorMenuBar::showMenuWindow()
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

void EditorMenuBar::showMenuHelp()
{
	if (ImGui::MenuItem("About PhysicsEngine")) {
		aboutClicked = true;
	}
}