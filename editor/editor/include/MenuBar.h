#ifndef __MAIN_MENU_BAR_H__
#define __MAIN_MENU_BAR_H__

#include <string>

#include "EditorClipboard.h"
#include "Filebrowser.h"

#include "views/ProjectWindow.h"
#include "views/PreferencesWindow.h"
#include "views/AboutPopup.h"
#include "views/BuildWindow.h"

namespace PhysicsEditor
{
class MenuBar
{
  private:
    // File
    bool newSceneClicked;
    bool openSceneClicked;
    bool saveClicked;
    bool saveAsClicked;
    bool newProjectClicked;
    bool openProjectClicked;
    bool saveProjectClicked;
    bool buildClicked;
    bool quitClicked;

    // Edit
    bool preferencesClicked;
    bool runTestsClicked;

    // Windows
    bool openInspectorClicked;
    bool openHierarchyClicked;
    bool openConsoleClicked;
    bool openSceneViewClicked;
    bool openProjectViewClicked;

    // About
    bool aboutClicked;

    Filebrowser filebrowser;
    
    ProjectWindow projectWindow;
    BuildWindow buildWindow;
    PreferencesWindow preferencesWindow;
    AboutPopup aboutPopup;

  public:
    MenuBar();
    ~MenuBar();

    void init(EditorClipboard& clipboard);
    void update(EditorClipboard &clipboard);

    bool isNewSceneClicked() const;
    bool isOpenSceneClicked() const;
    bool isSaveClicked() const;
    bool isSaveAsClicked() const;
    bool isBuildClicked() const;
    bool isQuitClicked() const;
    bool isNewProjectClicked() const;
    bool isOpenProjectClicked() const;
    bool isSaveProjectClicked() const;
    bool isOpenInspectorCalled() const;
    bool isOpenHierarchyCalled() const;
    bool isOpenConsoleCalled() const;
    bool isOpenSceneViewCalled() const;
    bool isOpenProjectViewCalled() const;
    bool isAboutClicked() const;
    bool isPreferencesClicked() const;
    bool isRunTestsClicked() const;

  private:
    void showMenuFile(EditorClipboard& clipboard);
    void showMenuEdit(EditorClipboard& clipboard);
    void showMenuWindow(EditorClipboard& clipboard);
    void showMenuHelp(EditorClipboard& clipboard);

    void newScene(EditorClipboard& clipboard);
    void openScene(EditorClipboard& clipboard, std::string name, std::string path);
    void saveScene(EditorClipboard& clipboard, std::string name, std::string path);
    void newProject(EditorClipboard& clipboard);
    void openProject(EditorClipboard& clipboard);
    void saveProject(EditorClipboard& clipboard);
    void build(EditorClipboard& clipboard);
};
} // namespace PhysicsEditor

#endif