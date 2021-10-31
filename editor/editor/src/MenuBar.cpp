#include "../include/MenuBar.h"
#include "../include/EditorSceneManager.h"
#include "../include/EditorProjectManager.h"

#include "imgui.h"

using namespace PhysicsEditor;

MenuBar::MenuBar()
{
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
    runTestsClicked = false;
}

MenuBar::~MenuBar()
{
}

void MenuBar::init(Clipboard &clipboard)
{
    aboutPopup.init(clipboard);
    preferencesWindow.init(clipboard);
    buildWindow.init(clipboard);
}

void MenuBar::update(Clipboard &clipboard)
{
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
    runTestsClicked = false;

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            showMenuFile(clipboard);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit"))
        {
            showMenuEdit(clipboard);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Windows"))
        {
            showMenuWindow(clipboard);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help"))
        {
            showMenuHelp(clipboard);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    // new, open and save scene
    if (isNewSceneClicked())
    {
        newScene(clipboard);
    }
    if (isOpenSceneClicked())
    {
        filebrowser.setMode(FilebrowserMode::Open);
    }
    else if (isSaveClicked() && !clipboard.getScenePath().empty())
    {
        saveScene(clipboard, clipboard.getSceneName(), clipboard.getScenePath());
    }
    else if (isSaveAsClicked() || isSaveClicked() && clipboard.getScenePath().empty())
    {
        filebrowser.setMode(FilebrowserMode::Save);
    }

    filebrowser.render(clipboard.getProjectPath(),
                       isOpenSceneClicked() || isSaveAsClicked() || isSaveClicked() && clipboard.getScenePath().empty());

    if (filebrowser.isOpenClicked())
    {
        openScene(clipboard, filebrowser.getOpenFilePath().filename().string(), filebrowser.getOpenFilePath());
    }
    else if (filebrowser.isSaveClicked())
    {
        saveScene(clipboard, filebrowser.getSaveFilePath().filename().string(), filebrowser.getSaveFilePath());
    }

    // new, open, save project project
    if (isOpenProjectClicked())
    {
        projectWindow.setMode(ProjectWindowMode::OpenProject);
    }
    else if (isNewProjectClicked())
    {
        projectWindow.setMode(ProjectWindowMode::NewProject);
    }
    else if (isSaveProjectClicked())
    {
        saveProject(clipboard);
    }

    projectWindow.draw(clipboard, isOpenProjectClicked() || isNewProjectClicked());
    aboutPopup.draw(clipboard, isAboutClicked());
    preferencesWindow.draw(clipboard, isPreferencesClicked());
    buildWindow.draw(clipboard, isBuildClicked());
}

bool MenuBar::isNewSceneClicked() const
{
    return newSceneClicked;
}

bool MenuBar::isOpenSceneClicked() const
{
    return openSceneClicked;
}

bool MenuBar::isSaveClicked() const
{
    return saveClicked;
}

bool MenuBar::isSaveAsClicked() const
{
    return saveAsClicked;
}

bool MenuBar::isBuildClicked() const
{
    return buildClicked;
}

bool MenuBar::isQuitClicked() const
{
    return quitClicked;
}

bool MenuBar::isNewProjectClicked() const
{
    return newProjectClicked;
}

bool MenuBar::isOpenProjectClicked() const
{
    return openProjectClicked;
}

bool MenuBar::isSaveProjectClicked() const
{
    return saveProjectClicked;
}

bool MenuBar::isOpenInspectorCalled() const
{
    return openInspectorClicked;
}

bool MenuBar::isOpenHierarchyCalled() const
{
    return openHierarchyClicked;
}

bool MenuBar::isOpenConsoleCalled() const
{
    return openConsoleClicked;
}

bool MenuBar::isOpenSceneViewCalled() const
{
    return openSceneViewClicked;
}

bool MenuBar::isOpenProjectViewCalled() const
{
    return openProjectViewClicked;
}

bool MenuBar::isAboutClicked() const
{
    return aboutClicked;
}

bool MenuBar::isPreferencesClicked() const
{
    return preferencesClicked;
}

bool MenuBar::isRunTestsClicked() const
{
    return runTestsClicked;
}

void MenuBar::showMenuFile(const Clipboard &clipboard)
{
    if (ImGui::MenuItem("New Scene", NULL, false, !clipboard.getProjectPath().empty()))
    {
        newSceneClicked = true;
    }
    if (ImGui::MenuItem("Open Scene", "Ctrl+O", false, !clipboard.getProjectPath().empty()))
    {
        openSceneClicked = true;
    }

    ImGui::Separator();

    if (ImGui::MenuItem("Save Scene", "Ctrl+S", false, clipboard.getSceneId().isValid()))
    {
        saveClicked = true;
    }
    if (ImGui::MenuItem("Save As..", nullptr, false, clipboard.getSceneId().isValid()))
    {
        saveAsClicked = true;
    }

    ImGui::Separator();

    if (ImGui::MenuItem("New Project"))
    {
        newProjectClicked = true;
    }
    if (ImGui::MenuItem("Open Project"))
    {
        openProjectClicked = true;
    }
    if (ImGui::MenuItem("Save Project", nullptr, false, !clipboard.getProjectPath().empty()))
    {
        saveProjectClicked = true;
    }
    if (ImGui::MenuItem("Build", nullptr, false, !clipboard.getProjectPath().empty()))
    {
        buildClicked = true;
    }

    if (ImGui::MenuItem("Quit", "Alt+F4"))
    {
        quitClicked = true;
    }
}

void MenuBar::showMenuEdit(const Clipboard &clipboard)
{
    //if (ImGui::MenuItem("Undo", "CTRL+Z", false, Undo::canUndo()))
    //{
    //    Undo::undoCommand();
    //}
    //if (ImGui::MenuItem("Redo", "CTRL+Y", false, Undo::canRedo()))
    //{
    //    Undo::executeCommand();
    //}
    ImGui::Separator();
    if (ImGui::MenuItem("Cut", "CTRL+X"))
    {
    }
    if (ImGui::MenuItem("Copy", "CTRL+C"))
    {
    }
    if (ImGui::MenuItem("Paste", "CTRL+V"))
    {
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Preferences..."))
    {
        preferencesClicked = true;
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Run Tests"))
    {
        runTestsClicked = true;
    }
}

void MenuBar::showMenuWindow(const Clipboard &clipboard)
{
    if (ImGui::MenuItem("Heirarchy"))
    {
        openHierarchyClicked = true;
    }
    if (ImGui::MenuItem("Inspector"))
    {
        openInspectorClicked = true;
    }
    if (ImGui::MenuItem("Console"))
    {
        openConsoleClicked = true;
    }
    if (ImGui::MenuItem("Scene View"))
    {
        openSceneViewClicked = true;
    }
    if (ImGui::MenuItem("Project View"))
    {
        openProjectViewClicked = true;
    }
}

void MenuBar::showMenuHelp(const Clipboard &clipboard)
{
    if (ImGui::MenuItem("About PhysicsEngine"))
    {
        aboutClicked = true;
    }
}

void MenuBar::newScene(Clipboard &clipboard)
{
    EditorSceneManager::newScene(clipboard);
}

void MenuBar::openScene(Clipboard& clipboard, const std::string& name, const std::filesystem::path& path)
{
    EditorSceneManager::openScene(clipboard, name, path);
}

void MenuBar::saveScene(Clipboard& clipboard, const std::string& name, const std::filesystem::path& path)
{
    EditorSceneManager::saveScene(clipboard, name, path);
}

void MenuBar::newProject(Clipboard &clipboard)
{
    EditorProjectManager::newProject(clipboard);
}

void MenuBar::openProject(Clipboard &clipboard)
{
    EditorProjectManager::openProject(clipboard);
}

void MenuBar::saveProject(Clipboard &clipboard)
{
    EditorProjectManager::saveProject(clipboard);
}

void MenuBar::build(Clipboard &clipboard)
{
}