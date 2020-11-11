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

#include "../include/imgui_extensions.h"
#include "../include/imgui_styles.h"

#include "core/WriteInternalToJson.h"

#include "core/InternalShaders.h"

using namespace PhysicsEditor;
using namespace json;

Editor::Editor()
{
    cameraSystem = NULL;
    renderSystem = NULL;
    cleanupSystem = NULL;

    currentProject = {};
    currentScene = {};
    clipboard = {};
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

    // add editor camera to world
    PhysicsEditor::createEditorCamera(&world, editorOnlyEntityIds);

    // add camera, render, and cleanup system to world
    cameraSystem = world.addSystem<EditorCameraSystem>(0);
    // add simple editor render pass system to render line floor and default skymap
    renderSystem = world.addSystem<RenderSystem>(1);
    // add gizmo system
    gizmoSystem = world.addSystem<GizmoSystem>(2);
    // add simple editor render system to render gizmo's
    cleanupSystem = world.addSystem<CleanUpSystem>(3);

    renderSystem->mRenderToScreen = false;

    for (int i = 0; i < world.getNumberOfUpdatingSystems(); i++)
    {
        System *system = world.getSystemByUpdateOrder(i);

        system->init(&world);
    }

    PhysicsEngine::Graphics::checkError(__LINE__, __FILE__);
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

    PhysicsEngine::Graphics::checkError(__LINE__, __FILE__);

    libraryDirectory.update();

    libraryDirectory.loadQueuedAssetsIntoWorld(&world);

    // start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    ImGui::ShowDemoWindow();

    // draw menu and toolbar
    editorMenu.render(currentProject, currentScene);
    editorToolbar.render(clipboard);

    // TODO: Should this be moved into editorMenu.render?
    updateProjectAndSceneState();

    // draw hierarchy window
    hierarchy.render(&world, currentScene, clipboard, editorOnlyEntityIds, editorMenu.isOpenHierarchyCalled());

    // draw inspector window
    inspector.render(&world, currentProject, currentScene, clipboard, editorMenu.isOpenInspectorCalled());

    // draw console window
    console.render(editorMenu.isOpenConsoleCalled());

    // draw project view window
    projectView.render(currentProject.path, libraryDirectory, clipboard, editorBecameActiveThisFrame,
                       editorMenu.isOpenProjectViewCalled());

    aboutPopup.render(editorMenu.isAboutClicked());
    preferencesWindow.render(editorMenu.isPreferencesClicked());

    // draw scene view window
    sceneView.render(&world, cameraSystem, clipboard, editorMenu.isOpenSceneViewCalled());

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
    return currentProject.path;
}

std::string Editor::getCurrentScenePath() const
{
    return currentScene.path;
}

void Editor::newScene()
{
    // mark any (non-editor) entities in currently opened scene to be latent destroyed
    world.latentDestroyEntitiesInWorld(); // need to destroy assets too!

    // re-centre editor camera to default position
    cameraSystem->resetCamera();

    // clear any dragged and selected items on clipboard
    clipboard.clearDraggedItem();
    clipboard.clearSelectedItem();

    currentScene.name = "default.scene";
    currentScene.path = "";
    currentScene.metaPath = "";
    currentScene.libraryPath = "";
    currentScene.sceneId = Guid::newGuid();
    currentScene.isDirty = true;
}

void Editor::openScene(std::string name, std::string path)
{
    // check to make sure the scene is part of the current project
    if (path.find(currentProject.path + "\\data\\") != 0)
    {
        std::string errorMessage =
            "Could not open scene " + path + " because it is not part of current project " + currentProject.path + "\n";
        Log::error(&errorMessage[0]);
        return;
    }

    // meta scene file path
    std::string sceneMetaFilePath = path.substr(0, path.find(".")) + ".json";

    // get guid from scene meta file
    Guid sceneId = PhysicsEditor::findGuidFromMetaFilePath(sceneMetaFilePath);

    // binary scene file path
    std::string binarySceneFilePath = currentProject.path + "\\library\\" + sceneId.toString() + ".sdata";

    // mark any (non-editor) entities in currently opened scene to be latent destroyed
    // TODO: Need todestroy assets too!
    world.latentDestroyEntitiesInWorld();

    // reset editor camera to default position
    cameraSystem->resetCamera();

    // clear any dragged and selected items on clipboard
    clipboard.clearDraggedItem();
    clipboard.clearSelectedItem();

    // load binary version of scene into world (ignoring systems and cameras)
    if (world.loadSceneFromEditor(binarySceneFilePath))
    {
        currentScene.name = name;
        currentScene.path = path;
        currentScene.metaPath = sceneMetaFilePath;
        currentScene.libraryPath = binarySceneFilePath;
        currentScene.sceneId = sceneId;
        currentScene.isDirty = false;
    }
    else
    {
        std::string errorMessage = "Failed to load scene " + binarySceneFilePath + " into world\n";
        Log::error(&errorMessage[0]);
    }
}

void Editor::saveScene(std::string name, std::string path)
{
    if (!currentScene.isDirty)
    {
        return;
    }

    if (PhysicsEditor::writeSceneToJson(&world, path, editorOnlyEntityIds))
    {
        currentScene.name = name;
        currentScene.path = path;
        currentScene.isDirty = false;
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
            currentProject.name = name;
            currentProject.path = path;
            currentProject.isDirty = false;

            currentScene.name = "";
            currentScene.path = "";
            currentScene.metaPath = "";
            currentScene.libraryPath = "";
            currentScene.sceneId = Guid::INVALID;
            currentScene.isDirty = false;

            SetWindowTextA(window, ("Physics Engine - " + currentProject.path).c_str());
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
    world.latentDestroyEntitiesInWorld(); // need to destroy assets too!

    // tell library directory which project to watch
    libraryDirectory.watch(path);

    // reset editor camera
    cameraSystem->resetCamera();
}

void Editor::openProject(std::string name, std::string path)
{
    currentProject.name = name;
    currentProject.path = path;
    currentProject.isDirty = false;

    currentScene.name = "";
    currentScene.path = "";
    currentScene.metaPath = "";
    currentScene.libraryPath = "";
    currentScene.sceneId = Guid::INVALID;
    currentScene.isDirty = false;

    // mark any (non-editor) entities in currently opened scene to be latent destroyed
    world.latentDestroyEntitiesInWorld();

    // tell library directory which project to watch
    libraryDirectory.watch(path);

    // reset editor camera
    cameraSystem->resetCamera();

    SetWindowTextA(window, ("Physics Engine - " + currentProject.path).c_str());
}

void Editor::saveProject(std::string name, std::string path)
{
    if (!currentProject.isDirty)
    {
        return;
    }

    for (int i = 0; i < world.getNumberOfAssets<Material>(); i++)
    {
        Material *material = world.getAssetByIndex<Material>(i);
        /*std::string assetPath = libraryDirectory.getFilePath(material->getId());

        if (!PhysicsEditor::writeAssetToJson(&world, assetPath, material->getId(), AssetType<Material>::type)) {
            std::string message = "Could not save material in project " + assetPath + "\n";
            Log::error(message.c_str());
            return;
        }*/
    }

    currentScene.name = name;
    currentScene.path = path;
    currentScene.isDirty = false;
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
    else if (editorMenu.isSaveClicked() && currentScene.path != "")
    {
        saveScene(currentScene.name, currentScene.path);
    }
    else if (editorMenu.isSaveAsClicked() || editorMenu.isSaveClicked() && currentScene.path == "")
    {
        filebrowser.setMode(FilebrowserMode::Save);
    }

    filebrowser.render(currentProject.path, editorMenu.isOpenSceneClicked() || editorMenu.isSaveAsClicked() ||
                                                editorMenu.isSaveClicked() && currentScene.path == "");

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
        saveProject(currentProject.name, currentProject.path);
    }

    projectWindow.render(editorMenu.isOpenProjectClicked() | editorMenu.isNewProjectClicked());

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