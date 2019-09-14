#include <fstream>

#include "../include/Editor.h"
#include "../include/FileSystemUtil.h"
#include "../include/EditorCameraSystem.h" // could just add this to the engine lib? Could include a couple different camera movement systems like editor, fps etc in engine as examples?

#include "core/Log.h"
#include "systems/RenderSystem.h"
#include "systems/CleanUpSystem.h"

#include <json/json.hpp>

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;
using namespace json;

Editor::Editor()
{
	quitCalled = false;
	camera = NULL;

	currentProjectPath = "";
	currentScenePath = "";
}

Editor::~Editor()
{

}

void Editor::init(HWND window, int width, int height)
{
	// Setup Dear ImGui binding
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	//Init Win32
	ImGui_ImplWin32_Init(window);

	//Init OpenGL Imgui Implementation
	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 130";
	ImGui_ImplOpenGL3_Init(glsl_version);

	//Set Window bg color
	ImVec4 clear_color = ImVec4(1.000F, 1.000F, 1.000F, 1.0F);

	// Setup style
	ImGui::StyleColorsClassic();
}

void Editor::cleanUp()
{
	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui::DestroyContext();
	ImGui_ImplWin32_Shutdown();
}

void Editor::render()
{
	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();

	ImGui::ShowDemoWindow();
	ImGui::Text(currentProjectPath.c_str());
	ImGui::Text(currentScenePath.c_str());

	libraryDirectory.update(currentProjectPath);

	updateAssetsLoadedInWorld();

	mainMenu.render();

	// new, open and save scene
	if (mainMenu.isNewClicked()){
		currentScenePath = "";
		newScene();
	}
	if (mainMenu.isOpenClicked()) {
		filebrowser.setMode(FilebrowserMode::Open);
	}
	else if (mainMenu.isSaveAsClicked()) {
		filebrowser.setMode(FilebrowserMode::Save);
	}

	filebrowser.render(mainMenu.isOpenClicked() | mainMenu.isSaveAsClicked());

	if (filebrowser.isOpenClicked()) {
		openScene(filebrowser.getOpenFilePath());
	}
	else if (filebrowser.isSaveClicked()) {
		saveScene(filebrowser.getSaveFilePath());
	}

	// new, open, save project project
	if (mainMenu.isOpenProjectClicked()) {
		projectWindow.setMode(ProjectWindowMode::OpenProject);
	}
	else if (mainMenu.isNewProjectClicked()) {
		projectWindow.setMode(ProjectWindowMode::NewProject);
	}

	projectWindow.render(mainMenu.isOpenProjectClicked() | mainMenu.isNewProjectClicked());

	if (projectWindow.isOpenClicked()) {
		openProject(projectWindow.getSelectedFolderPath());
	}
	else if (projectWindow.isCreateClicked()) {
		createProject(projectWindow.getSelectedFolderPath() + "\\" + projectWindow.getProjectName());
	}

	bool inspectorOpenedThisFrame = mainMenu.isOpenInspectorCalled();
	bool hierarchyOpenedThisFrame = mainMenu.isOpenHierarchyCalled();
	bool consoleOpenedThisFrame = mainMenu.isOpenConsoleCalled();

	hierarchy.render(world, hierarchyOpenedThisFrame);
	inspector.render(world, hierarchy.getSelectedEntity(), inspectorOpenedThisFrame);
	console.render(consoleOpenedThisFrame);

	aboutPopup.render(mainMenu.isAboutClicked());

	if (mainMenu.isQuitClicked()) {
		quitCalled = true;
	}

	// Rendering
	ImGui::Render();
	//wglMakeCurrent(deviceContext, renderContext);
	//glViewport(0, 0, g_display_w, g_display_h);                 //Display Size got from Resize Command
	glViewport(0, 0, 1920, 1080);
	/*glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);*/
	/*glClearColor(0.15f, 0.15f, 0.15f, 0.0f);*/
	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	Input input = {};

	for (int i = 0; i < world.getNumberOfSystems(); i++) {
		System* system = world.getSystemByIndex(i);

		system->update(input);
	}

	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	//wglMakeCurrent(deviceContext, renderContext);
	//SwapBuffers(deviceContext);
}

bool Editor::isQuitCalled() const
{
	return quitCalled;
}

std::string Editor::getCurrentProjectPath() const
{
	return currentProjectPath;
}

std::string Editor::getCurrentScenePath() const
{
	return currentScenePath;
}

void Editor::newScene()
{
	// mark any (non-editor) entities in currently opened scene to be latent destroyed
	for (int i = 0; i < world.getNumberOfEntities(); i++) {
		Entity* entity = world.getEntityByIndex(i);

		bool doNotDestroy = false;
		for (size_t j = 0; j < editorEntityIds.size(); j++) {
			if (editorEntityIds[j] == entity->entityId){
				doNotDestroy = true;
				break;
			}
		}

		if (!doNotDestroy){
			world.latentDestroyEntity(entity->entityId);
		}
	}

	// re-centre editor camera to default position
	camera->position = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->front = glm::vec3(1.0f, 0.0f, 0.0f);
	camera->up = glm::vec3(0.0f, 0.0f, 1.0f);
}

void Editor::openScene(std::string path)
{
	// clear world here like we do in newScene?

	// check to make sure the scene is part of the current project
	if (path.find(currentProjectPath + "\\data\\") != 0){
		std::string errorMessage = "Could not open scene " + path + " because it is not part of current project " + currentProjectPath + "\n";
		Log::error(&errorMessage[0]);
		return;
	}

	std::string metaFilePath = path.substr(0, path.find(".")) + ".json";

	// get guid from scene meta file
	std::ifstream metaFile(metaFilePath, std::ios::in);

	Guid guid;
	if (metaFile.is_open()) {
		std::ostringstream contents;
		contents << metaFile.rdbuf();
		metaFile.close();

		json::JSON jsonObject = JSON::Load(contents.str());
		guid = Guid(jsonObject["id"].ToString());
	}
	else {
		std::string errorMessage = "Could not open meta file " + metaFilePath + "\n";
		Log::error(&errorMessage[0]);
		return;
	}

	// get binary scene file from library directory
	std::string binarySceneFilePath = currentProjectPath + "\\library\\" + guid.toString() + ".data";

	// load binary version of scene into world
	if (world.loadScene(binarySceneFilePath)){
		currentScenePath = path;
	}
	else {
		std::string errorMessage = "Failed to load scene " + binarySceneFilePath + " into world\n";
		Log::error(&errorMessage[0]);
	}
}

void Editor::saveScene(std::string path)
{
	
}

void Editor::createProject(std::string path)
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

		if (success){
			currentProjectPath = path;
			assetsAddedToWorld.clear();
			Log::info("Project successfully createed\n");
		}
		else {
			Log::error("Could not create project sub directories\n");
		}
	}
	else {
		Log::error("Could not create project root directory\n");
	}

	// when creating new project or switching between projects, clear entire world?
	// when opening scenes within a project, only clear components and entities?


	// add editor camera to world
	//Entity* cameraEntity = world.createEntity();
	//camera = cameraEntity->addComponent<Camera>(&world);
	//camera->viewport.width = 1920;
	//camera->viewport.height = 1080;
	///*camera->position = glm::vec3(0.0f, 0.0f, 1.0f);
	//camera->front = glm::vec3(1.0f, 0.0f, 0.0f);
	//camera->up = glm::vec3(0.0f, 0.0f, 1.0f);*/

	//// add physics, render, and cleanup system to world
	//world.addSystem<EditorCameraSystem>(0);
	//world.addSystem<RenderSystem>(1);
	//world.addSystem<CleanUpSystem>(2);

	//for (int i = 0; i < world.getNumberOfSystems(); i++) {
	//	System* system = world.getSystemByIndex(i);

	//	system->init(&world);
	//}
}

void Editor::openProject(std::string path)
{
	currentProjectPath = path;

	assetsAddedToWorld.clear();


}

void Editor::updateAssetsLoadedInWorld()
{
	std::map<std::string, FileInfo> filePathToFileInfo = libraryDirectory.getTrackedFilesInProject();
	for (std::map<std::string, FileInfo>::iterator it1 = filePathToFileInfo.begin(); it1 != filePathToFileInfo.end(); it1++) {
		std::string filePath = it1->first;
		PhysicsEngine::Guid id = it1->second.id;
		std::string extension = it1->second.fileExtension;

		if (extension == "scene") {
			continue;
		}

		std::unordered_set<PhysicsEngine::Guid>::iterator it2 = assetsAddedToWorld.find(id);
		if (it2 == assetsAddedToWorld.end() ){
			assetsAddedToWorld.insert(id);

			// get file path of binary version of asset located in library directory
			std::string libraryFilePath = currentProjectPath + "\\library\\" + id.toString() + ".data";;

			if (!world.loadAsset(libraryFilePath)){
				std::string errorMessage = "Could not load asset: " + libraryFilePath + "\n";
				Log::error(&errorMessage[0]);
			}
		}
	}
}