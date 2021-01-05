#include <chrono>
#include <fstream>
#include <thread>

#include "../include/Editor.h"
#include "../include/EditorCameraSystem.h" // could just add this to the engine lib? Could include a couple different camera movement systems like editor, fps etc in engine as examples?
#include "../include/EditorFileIO.h"
#include "../include/EditorOnlyEntityCreation.h"
#include "../include/FileSystemUtil.h"

#include "components/Light.h"
#include "core/Log.h"
#include "systems/CleanUpSystem.h"
#include "systems/RenderSystem.h"

#include "graphics/Graphics.h"

#include <json/json.hpp>

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

#include "../include/imgui/imgui_extensions.h"
#include "../include/imgui/imgui_styles.h"

#include "core/WriteInternalToJson.h"

#include "core/InternalShaders.h"

#include "../include/IconsFontAwesome4.h"

using namespace PhysicsEditor;
using namespace json;

Editor::Editor()
{
}

Editor::~Editor()
{
}

void Editor::init(HWND window, int width, int height)
{
    this->window = window;

    // Setup Dear ImGui binding
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;

    // enable docking
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    // Init Win32
    ImGui_ImplWin32_Init(window);

    // Init OpenGL Imgui Implementation
    // GL 3.0 + GLSL 130
    ImGui_ImplOpenGL3_Init("#version 330");

    // Setup style
    ImGui::StyleColorsCorporate();

    io.Fonts->AddFontDefault();

    ImFontConfig config;
    config.MergeMode = true;
    config.GlyphMinAdvanceX = 13.0f; // Use if you want to make the icon monospaced
    static const ImWchar icon_ranges[] = {ICON_MIN_FA, ICON_MAX_FA, 0};
    io.Fonts->AddFontFromFileTTF("C:\\Users\\jsand\\Downloads\\fontawesome-webfont.ttf", 13.0f, &config, icon_ranges);
    io.Fonts->Build();

    clipboard.init();

    aboutPopup.init(clipboard);

    hierarchy.init(clipboard);
    inspector.init(clipboard);
    console.init(clipboard);
    projectView.init(clipboard);
    sceneView.init(clipboard);
}

void Editor::cleanUp()
{
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();
    ImGui_ImplWin32_Shutdown();
}

void Editor::render(bool editorBecameActiveThisFrame)
{
    // ImGui::ShowDemoWindow();
    // ImGui::ShowMetricsWindow();
    // ImGui::ShowStyleEditor();

    clipboard.getLibrary().update();
    clipboard.getLibrary().loadQueuedAssetsIntoWorld(clipboard.getWorld());

    // start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    ImGui::ShowDemoWindow();

    // draw menu and toolbar
    editorMenu.render(clipboard);
    editorToolbar.render(clipboard);

    // TODO: Should this be moved into editorMenu.render?
    updateProjectAndSceneState();

    hierarchy.update(clipboard, editorMenu.isOpenHierarchyCalled());
    inspector.update(clipboard, editorMenu.isOpenInspectorCalled());
    console.update(clipboard, editorMenu.isOpenConsoleCalled());
    projectView.update(clipboard, editorBecameActiveThisFrame, editorMenu.isOpenProjectViewCalled());

    aboutPopup.update(clipboard, editorMenu.isAboutClicked());
    preferencesWindow.update(clipboard, editorMenu.isPreferencesClicked());

    sceneView.update(clipboard, editorMenu.isOpenSceneViewCalled());

    // imgui render calls
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    ImGui::EndFrame();

    commandManager.update();
}

bool Editor::isQuitCalled() const
{
    return editorMenu.isQuitClicked();
}

std::string Editor::getCurrentProjectPath() const
{
    return clipboard.getProjectPath();
}

std::string Editor::getCurrentScenePath() const
{
    return clipboard.getScenePath();
}

void Editor::newScene()
{
    // mark any (non-editor) entities in currently opened scene to be latent destroyed
    clipboard.getWorld()->latentDestroyEntitiesInWorld(); // need to destroy assets too!

    // re-centre editor camera to default position
    clipboard.getWorld()->getSystem<EditorCameraSystem>()->resetCamera();

    // clear any dragged and selected items on clipboard
    clipboard.clearDraggedItem();
    clipboard.clearSelectedItem();

    clipboard.openScene("default.scene", "", "", "", Guid::newGuid());
}

void Editor::openScene(std::string name, std::string path)
{
    // check to make sure the scene is part of the current project
    if (path.find(clipboard.getProjectPath() + "\\data\\") != 0)
    {
        std::string errorMessage = "Could not open scene " + path + " because it is not part of current project " +
                                   clipboard.getProjectPath() + "\n";
        Log::error(&errorMessage[0]);
        return;
    }

    // meta scene file path
    std::string sceneMetaFilePath = path.substr(0, path.find(".")) + ".json";

    // get guid from scene meta file
    Guid sceneId = PhysicsEditor::findGuidFromMetaFilePath(sceneMetaFilePath);

    // binary scene file path
    std::string binarySceneFilePath = clipboard.getProjectPath() + "\\library\\" + sceneId.toString() + ".sdata";

    // mark any (non-editor) entities in currently opened scene to be latent destroyed
    // TODO: Need todestroy assets too!
    clipboard.getWorld()->latentDestroyEntitiesInWorld();

    // reset editor camera to default position
    clipboard.getWorld()->getSystem<EditorCameraSystem>()->resetCamera();

    // clear any dragged and selected items on clipboard
    clipboard.clearDraggedItem();
    clipboard.clearSelectedItem();

    // load binary version of scene into world (ignoring systems and cameras)
    if (clipboard.getWorld()->loadSceneFromEditor(binarySceneFilePath))
    {
        clipboard.openScene(name, path, sceneMetaFilePath, binarySceneFilePath, sceneId);
    }
    else
    {
        std::string errorMessage = "Failed to load scene " + binarySceneFilePath + " into world\n";
        Log::error(&errorMessage[0]);
    }
}

void Editor::saveScene(std::string name, std::string path)
{
    // if (!currentScene.isDirty)
    //{
    //    return;
    //}

    if (PhysicsEditor::writeSceneToJson(clipboard.getWorld(), path, clipboard.getEditorOnlyIds()))
    {
        clipboard.openScene(name, path);
    }
    else
    {
        std::string message = "Could not save world to scene file " + path + "\n";
        Log::error(message.c_str());
        return;
    }
}

void Editor::createProject(std::string name, std::string path)
{
    if (PhysicsEditor::createDirectory(path))
    {
        bool success = true;
        success &= createDirectory(path + "\\data");
        success &= createDirectory(path + "\\data\\scenes");
        success &= createDirectory(path + "\\data\\textures");
        success &= createDirectory(path + "\\data\\meshes");
        success &= createDirectory(path + "\\data\\materials");
        success &= createDirectory(path + "\\data\\shaders");

        if (success)
        {
            clipboard.openProject(name, path);
            clipboard.openScene("", "", "", "", Guid::INVALID);

            SetWindowTextA(window, ("Physics Engine - " + clipboard.getProjectPath()).c_str());
        }
        else
        {
            Log::error("Could not create project sub directories\n");
            return;
        }
    }
    else
    {
        Log::error("Could not create project root directory\n");
        return;
    }

    // mark any (non-editor) entities in currently opened scene to be latent destroyed
    clipboard.getWorld()->latentDestroyEntitiesInWorld();

    // tell library directory which project to watch
    clipboard.getLibrary().watch(path);
    ;

    // reset editor camera
    clipboard.getWorld()->getSystem<EditorCameraSystem>()->resetCamera();
}

void Editor::openProject(std::string name, std::string path)
{
    clipboard.openProject(name, path);
    clipboard.openScene("", "", "", "", Guid::INVALID);

    // mark any (non-editor) entities in currently opened scene to be latent destroyed
    clipboard.getWorld()->latentDestroyEntitiesInWorld();

    // tell library directory which project to watch
    clipboard.getLibrary().watch(path);

    // reset editor camera
    clipboard.getWorld()->getSystem<EditorCameraSystem>()->resetCamera();

    SetWindowTextA(window, ("Physics Engine - " + clipboard.getProjectPath()).c_str());
}

void Editor::saveProject(std::string name, std::string path)
{
    // if (!currentProject.isDirty)
    //{
    //    return;
    //}

    clipboard.openScene(name, path);
}

void Editor::updateProjectAndSceneState()
{
    // new, open and save scene
    if (editorMenu.isNewSceneClicked())
    {
        newScene();
    }
    if (editorMenu.isOpenSceneClicked())
    {
        filebrowser.setMode(FilebrowserMode::Open);
    }
    else if (editorMenu.isSaveClicked() && clipboard.getScenePath() != "")
    {
        saveScene(clipboard.getScene(), clipboard.getScenePath());
    }
    else if (editorMenu.isSaveAsClicked() || editorMenu.isSaveClicked() && clipboard.getScenePath() == "")
    {
        filebrowser.setMode(FilebrowserMode::Save);
    }

    filebrowser.render(clipboard.getProjectPath(), editorMenu.isOpenSceneClicked() || editorMenu.isSaveAsClicked() ||
                                                       editorMenu.isSaveClicked() && clipboard.getScenePath() == "");

    if (filebrowser.isOpenClicked())
    {
        openScene(filebrowser.getOpenFile(), filebrowser.getOpenFilePath());
    }
    else if (filebrowser.isSaveClicked())
    {
        saveScene(filebrowser.getSaveFile(), filebrowser.getSaveFilePath());
    }

    // new, open, save project project
    if (editorMenu.isOpenProjectClicked())
    {
        projectWindow.setMode(ProjectWindowMode::OpenProject);
    }
    else if (editorMenu.isNewProjectClicked())
    {
        projectWindow.setMode(ProjectWindowMode::NewProject);
    }
    else if (editorMenu.isSaveProjectClicked())
    {
        saveProject(clipboard.getProject(), clipboard.getProjectPath());
    }

    projectWindow.update(clipboard, editorMenu.isOpenProjectClicked() | editorMenu.isNewProjectClicked());

    if (projectWindow.isOpenClicked())
    {
        openProject(projectWindow.getProjectName(), projectWindow.getSelectedFolderPath());
    }
    else if (projectWindow.isCreateClicked())
    {
        createProject(projectWindow.getProjectName(),
                      projectWindow.getSelectedFolderPath() + "\\" + projectWindow.getProjectName());
    }
}