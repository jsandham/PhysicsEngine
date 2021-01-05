#ifndef __EDITOR_H__
#define __EDITOR_H__

#include <string>
#include <unordered_set>
#include <windows.h>

#include "components/Camera.h"
#include "core/Input.h"
#include "core/Time.h"
#include "core/World.h"

#include "LibraryDirectory.h"

#include "views/AboutPopup.h"
#include "views/BuildWindow.h"
#include "views/Console.h"
#include "views/Hierarchy.h"
#include "views/Inspector.h"
#include "views/PreferencesWindow.h"
#include "views/ProjectView.h"
#include "views/ProjectWindow.h"
#include "views/SceneView.h"

#include "CommandManager.h"
#include "EditorClipboard.h"
#include "EditorMenuBar.h"
#include "EditorToolbar.h"
#include "Filebrowser.h"

#include "EditorCameraSystem.h"
#include "systems/CleanUpSystem.h"
#include "systems/RenderSystem.h"
#include "systems/GizmoSystem.h"

namespace PhysicsEditor
{
class Editor
{
  private:
    HWND window;

    //LibraryDirectory libraryDirectory;
    CommandManager commandManager;

    EditorMenuBar editorMenu;
    EditorToolbar editorToolbar;
    Inspector inspector;
    Hierarchy hierarchy;
    ProjectView projectView;
    Console console;
    SceneView sceneView;
    Filebrowser filebrowser;
    ProjectWindow projectWindow; // ProjectBrowser? ProjectPopup?
    BuildWindow buildWindow;
    PreferencesWindow preferencesWindow;
    AboutPopup aboutPopup;

    EditorClipboard clipboard;

  public:
    Editor();
    ~Editor();

    void init(HWND window, int width, int height);
    void cleanUp();
    void render(bool editorApplicationActive);

    bool isQuitCalled() const;
    std::string getCurrentProjectPath() const;
    std::string getCurrentScenePath() const;

  private:
    void newScene();
    void openScene(std::string name, std::string path);
    void saveScene(std::string name, std::string path);
    void createProject(std::string name, std::string path);
    void openProject(std::string name, std::string path);
    void saveProject(std::string name, std::string path);
    void updateProjectAndSceneState();
};
} // namespace PhysicsEditor

#endif