#include "../include/MenuBar.h"
#include "../include/Undo.h"
#include "../include/EditorCameraSystem.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;
using namespace PhysicsEngine;

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
        openScene(clipboard, filebrowser.getOpenFile(), filebrowser.getOpenFilePath());
    }
    else if (filebrowser.isSaveClicked())
    {
        saveScene(clipboard, filebrowser.getSaveFile(), filebrowser.getSaveFilePath());
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
        // saveProject(clipboard, clipboard.getProject(), clipboard.getProjectPath());
    }

    projectWindow.draw(clipboard, isOpenProjectClicked() | isNewProjectClicked());
    aboutPopup.draw(clipboard, isAboutClicked());
    preferencesWindow.draw(clipboard, isPreferencesClicked());
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

    if (ImGui::MenuItem("Save Scene", "Ctrl+S", false, !clipboard.getScenePath().empty()))
    {
        saveClicked = true;
    }
    if (ImGui::MenuItem("Save As..", NULL, false, clipboard.getSceneId().isValid()))
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
    if (ImGui::MenuItem("Save Project", NULL, false, !clipboard.getProjectPath().empty()))
    {
        saveProjectClicked = true;
    }
    if (ImGui::MenuItem("Build", NULL, false, !clipboard.getProjectPath().empty()))
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
    if (ImGui::MenuItem("Undo", "CTRL+Z", false, Undo::canUndo()))
    {
        Undo::undoCommand();
    }
    if (ImGui::MenuItem("Redo", "CTRL+Y", false, Undo::canRedo()))
    {
        Undo::executeCommand();
    }
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
    // mark any (non-editor) entities in currently opened scene to be latent destroyed
    clipboard.getWorld()->latentDestroyEntitiesInWorld();

    // re-centre editor camera to default position
    clipboard.getWorld()->getSystem<EditorCameraSystem>()->resetCamera();

    // clear any dragged and selected items on clipboard
    clipboard.clearDraggedItem();
    clipboard.clearSelectedItem();

    Scene* scene = clipboard.getWorld()->createScene();
    if (scene != nullptr)
    {
        clipboard.setActiveScene("default.scene", "", scene->getId());
    }
}

void MenuBar::openScene(Clipboard& clipboard, const std::string& name, const std::string& path)
{
    // check to make sure the scene is part of the current project
    if (path.find(clipboard.getProjectPath() + "\\data\\") != 0)
    {
        return;
    }

    // mark any (non-editor) entities in currently opened scene to be latent destroyed
    /*clipboard.getWorld()->latentDestroyEntitiesInWorld();*/
    clipboard.getWorld()->immediateDestroyEntitiesInWorld();

    // reset editor camera to default position
    clipboard.getWorld()->getSystem<EditorCameraSystem>()->resetCamera();

    // clear any dragged and selected items on clipboard
    clipboard.clearDraggedItem();
    clipboard.clearSelectedItem();

    // load scene into world
    Scene* scene = clipboard.getWorld()->loadSceneFromYAML(path);
    if (scene != nullptr)
    {
        clipboard.setActiveScene(name, path, scene->getId());
    }
}

void MenuBar::saveScene(Clipboard& clipboard, const std::string& name, const std::string& path)
{
    clipboard.getWorld()->writeSceneToYAML(path, clipboard.getSceneId());
}

void MenuBar::newProject(Clipboard &clipboard)
{
}

void MenuBar::openProject(Clipboard &clipboard)
{
}

void MenuBar::saveProject(Clipboard &clipboard)
{
}

void MenuBar::build(Clipboard &clipboard)
{
}