#include "../include/MenuBar.h"
#include "../include/ProjectDatabase.h"

#include "imgui.h"

using namespace PhysicsEditor;

extern std::string openFile(const char* filter);
extern std::string saveFile(const char* filter);
extern std::string selectFolder();

MenuBar::MenuBar()
{
    mNewSceneClicked = false;
    mOpenSceneClicked = false;
    mSaveClicked = false;
    mSaveAsClicked = false;
    mNewProjectClicked = false;
    mOpenProjectClicked = false;
    mSaveProjectClicked = false;
    mBuildClicked = false;
    mQuitClicked = false;
    mOpenInspectorClicked = false;
    mOpenHierarchyClicked = false;
    mOpenConsoleClicked = false;
    mOpenSceneViewClicked = false;
    mOpenProjectViewClicked = false;
    mAboutClicked = false;
    mPreferencesClicked = false;
    mPopulateTestScene = false;
}

MenuBar::~MenuBar()
{
}

void MenuBar::init(Clipboard &clipboard)
{
    mAboutPopup.init(clipboard);
    mPreferencesWindow.init(clipboard);
    mBuildWindow.init(clipboard);
}

void MenuBar::update(Clipboard &clipboard)
{
    mNewSceneClicked = false;
    mOpenSceneClicked = false;
    mSaveClicked = false;
    mSaveAsClicked = false;
    mNewProjectClicked = false;
    mOpenProjectClicked = false;
    mSaveProjectClicked = false;
    mBuildClicked = false;
    mQuitClicked = false;
    mOpenInspectorClicked = false;
    mOpenHierarchyClicked = false;
    mOpenConsoleClicked = false;
    mOpenSceneViewClicked = false;
    mOpenProjectViewClicked = false;
    mAboutClicked = false;
    mPreferencesClicked = false;
    mPopulateTestScene = false;

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
        ProjectDatabase::newScene(clipboard, "default.scene");
    }

    if (isOpenSceneClicked())
    {
        std::string fileToOpen = openFile("Scene (*.scene)\0*.scene\0");
        if (!fileToOpen.empty())
        {
            ProjectDatabase::openScene(clipboard, fileToOpen);
        }
    }
    else if (isSaveClicked())
    {
        ProjectDatabase::saveScene(clipboard, clipboard.getScenePath());
    }
    else if (isSaveAsClicked())
    {
        std::string fileToSave = saveFile("Scene (*.scene)\0*.scene\0");
        if (!fileToSave.empty())
        {
            ProjectDatabase::openScene(clipboard, fileToSave);
        }
    }

    // new, open, save project project
    if (isOpenProjectClicked())
    {
        std::string folderToOpen = selectFolder();
        if (!folderToOpen.empty())
        {
            ProjectDatabase::openProject(clipboard, folderToOpen);
        }
    }
    else if (isSaveProjectClicked())
    {
        ProjectDatabase::saveProject(clipboard);
    }

    if (isPopulateTestSceneClicked())
    {
        ProjectDatabase::populateScene(clipboard);
    }

    mProjectWindow.draw(clipboard, isNewProjectClicked());
    mAboutPopup.draw(clipboard, isAboutClicked());
    mPreferencesWindow.draw(clipboard, isPreferencesClicked());
    mBuildWindow.draw(clipboard, isBuildClicked());
}

bool MenuBar::isNewSceneClicked() const
{
    return mNewSceneClicked;
}

bool MenuBar::isOpenSceneClicked() const
{
    return mOpenSceneClicked;
}

bool MenuBar::isSaveClicked() const
{
    return mSaveClicked;
}

bool MenuBar::isSaveAsClicked() const
{
    return mSaveAsClicked;
}

bool MenuBar::isBuildClicked() const
{
    return mBuildClicked;
}

bool MenuBar::isQuitClicked() const
{
    return mQuitClicked;
}

bool MenuBar::isNewProjectClicked() const
{
    return mNewProjectClicked;
}

bool MenuBar::isOpenProjectClicked() const
{
    return mOpenProjectClicked;
}

bool MenuBar::isSaveProjectClicked() const
{
    return mSaveProjectClicked;
}

bool MenuBar::isOpenInspectorCalled() const
{
    return mOpenInspectorClicked;
}

bool MenuBar::isOpenHierarchyCalled() const
{
    return mOpenHierarchyClicked;
}

bool MenuBar::isOpenConsoleCalled() const
{
    return mOpenConsoleClicked;
}

bool MenuBar::isOpenSceneViewCalled() const
{
    return mOpenSceneViewClicked;
}

bool MenuBar::isOpenProjectViewCalled() const
{
    return mOpenProjectViewClicked;
}

bool MenuBar::isAboutClicked() const
{
    return mAboutClicked;
}

bool MenuBar::isPreferencesClicked() const
{
    return mPreferencesClicked;
}

bool MenuBar::isPopulateTestSceneClicked() const
{
    return mPopulateTestScene;
}

void MenuBar::showMenuFile(const Clipboard &clipboard)
{
    if (ImGui::MenuItem("New Scene", NULL, false, !clipboard.getProjectPath().empty()))
    {
        mNewSceneClicked = true;
    }
    if (ImGui::MenuItem("Open Scene", "Ctrl+O", false, !clipboard.getProjectPath().empty()))
    {
        mOpenSceneClicked = true;
    }

    ImGui::Separator();

    if (ImGui::MenuItem("Save Scene", "Ctrl+S", false, clipboard.getSceneId().isValid()))
    {
        // if we dont have a scene path then it must be a new unsaved scene, call save as instead
        if (clipboard.getScenePath().empty())
        {
            mSaveAsClicked = true;
        }
        else
        {
            mSaveClicked = true;
        }
    }
    if (ImGui::MenuItem("Save As..", nullptr, false, clipboard.getSceneId().isValid()))
    {
        mSaveAsClicked = true;
    }

    ImGui::Separator();

    if (ImGui::MenuItem("New Project"))
    {
        mNewProjectClicked = true;
    }
    if (ImGui::MenuItem("Open Project"))
    {
        mOpenProjectClicked = true;
    }
    if (ImGui::MenuItem("Save Project", nullptr, false, !clipboard.getProjectPath().empty()))
    {
        mSaveProjectClicked = true;
    }
    if (ImGui::MenuItem("Build", nullptr, false, !clipboard.getProjectPath().empty()))
    {
        mBuildClicked = true;
    }

    if (ImGui::MenuItem("Quit", "Alt+F4"))
    {
        mQuitClicked = true;
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
        mPreferencesClicked = true;
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Populate Scene", nullptr, false, clipboard.getSceneId().isValid()))
    {
        mPopulateTestScene = true;
    }
}

void MenuBar::showMenuWindow(const Clipboard &clipboard)
{
    if (ImGui::MenuItem("Heirarchy"))
    {
        mOpenHierarchyClicked = true;
    }
    if (ImGui::MenuItem("Inspector"))
    {
        mOpenInspectorClicked = true;
    }
    if (ImGui::MenuItem("Console"))
    {
        mOpenConsoleClicked = true;
    }
    if (ImGui::MenuItem("Scene View"))
    {
        mOpenSceneViewClicked = true;
    }
    if (ImGui::MenuItem("Project View"))
    {
        mOpenProjectViewClicked = true;
    }
}

void MenuBar::showMenuHelp(const Clipboard &clipboard)
{
    if (ImGui::MenuItem("About PhysicsEngine"))
    {
        mAboutClicked = true;
    }
}