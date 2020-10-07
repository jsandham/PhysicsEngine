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

#include "AboutPopup.h"
#include "BuildWindow.h"
#include "CommandManager.h"
#include "Console.h"
#include "EditorClipboard.h"
#include "EditorMenuBar.h"
#include "EditorProject.h"
#include "EditorScene.h"
#include "EditorToolbar.h"
#include "Filebrowser.h"
#include "Hierarchy.h"
#include "Inspector.h"
#include "PreferencesWindow.h"
#include "ProjectView.h"
#include "ProjectWindow.h"
#include "SceneView.h"

#include "EditorCameraSystem.h"
#include "systems/CleanUpSystem.h"
#include "systems/RenderSystem.h"

namespace PhysicsEditor
{
class Editor
{
  private:
    HWND window;

    World world;

    LibraryDirectory libraryDirectory;
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

    EditorProject currentProject;
    EditorScene currentScene;
    EditorClipboard clipboard;

    EditorCameraSystem *cameraSystem;
    RenderSystem *renderSystem;
    CleanUpSystem *cleanupSystem;

    std::set<PhysicsEngine::Guid> editorOnlyEntityIds;

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