#include <fstream>

#include "../include/Editor.h"
#include "../include/FileSystemUtil.h"
#include "../include/EditorCameraSystem.h" // could just add this to the engine lib? Could include a couple different camera movement systems like editor, fps etc in engine as examples?

#include "core/Log.h"
#include "components/Light.h"
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

	// add physics, render, and cleanup system to world
	world.addSystem<EditorCameraSystem>(0);
	renderSystem = world.addSystem<RenderSystem>(1);
	world.addSystem<CleanUpSystem>(2);

	for (int i = 0; i < world.getNumberOfSystems(); i++) {
		System* system = world.getSystemByIndex(i);

		system->init(&world);
	}
	
	input = {};
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

	//ImGui::ShowDemoWindow();
	ImGui::ShowMetricsWindow();
	//ImGui::Text(currentProjectPath.c_str());
	//ImGui::Text(currentScenePath.c_str());

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

	glViewport(0, 0, 1920, 1080);
	glClearColor(1.0f, 0.412f, 0.706f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	updateInputPassedToSystems(&input);

	for (int i = 0; i < world.getNumberOfSystems(); i++) {
		System* system = world.getSystemByIndex(i);

		system->update(input);
	}

	ImGui::Begin("Scene");
	{
		ImGui::Image((void*)(intptr_t)renderSystem->getColorTexture(), ImVec2(1024, 1024));
	}
	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	ImGui::EndFrame();
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



	world.latentDestroyEntitiesInWorld(); // need to destroy assets too!

	// add editor camera to world
	/*Entity* cameraEntity = world.createEntity();
	camera = cameraEntity->addComponent<Camera>(&world);
	camera->viewport.width = 1024;
	camera->viewport.height = 1024;
	camera->position = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->front = glm::vec3(1.0f, 0.0f, 0.0f);
	camera->up = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->backgroundColor = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
	camera->updateInternalCameraState();
*/




	// load binary version of scene into world
	if (world.loadSceneFromEditor(binarySceneFilePath)){
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
			Log::info("Project successfully created\n");
		}
		else {
			Log::error("Could not create project sub directories\n");
			return;
		}
	}
	else {
		Log::error("Could not create project root directory\n");
		return;
	}


	world.latentDestroyEntitiesInWorld(); // need to destroy assets too!

	// add editor camera to world
	/*Entity* cameraEntity = world.createEntity();
	camera = cameraEntity->addComponent<Camera>(&world);
	camera->viewport.width = 1024;
	camera->viewport.height = 1024;
	camera->position = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->front = glm::vec3(1.0f, 0.0f, 0.0f);
	camera->up = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->updateInternalCameraState();*/



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

	world.latentDestroyEntitiesInWorld(); // need to destroy assets too!

	// add editor camera to world
	/*Entity* cameraEntity = world.createEntity();
	camera = cameraEntity->addComponent<Camera>(&world);
	camera->viewport.width = 1024;
	camera->viewport.height = 1024;
	camera->position = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->front = glm::vec3(1.0f, 0.0f, 0.0f);
	camera->up = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->updateInternalCameraState();*/
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

void Editor::updateInputPassedToSystems(Input* input)
{
	ImGuiIO& io = ImGui::GetIO();

	if (!io.WantCaptureKeyboard) {
		// capture input for main application
	}


	/*for (int i = 0; i < IM_ARRAYSIZE(io.KeysDown); i++) {
		ImGui::Text("i: %d %d\n", i, io.KeysDown[i]);
	}
*/
	/*struct Input
	{
		bool keyIsDown[51];
		bool keyWasDown[51];
		bool mouseButtonIsDown[3];
		bool mouseButtonWasDown[3];
		bool xboxButtonIsDown[14];
		bool xboxButtonWasDown[14];
		int mousePosX;
		int mousePosY;
		int mouseDelta;
		int leftStickX;
		int leftStickY;
		int rightStickX;
		int rightStickY;
	};*/

	// keyboard keys
	//io.KeyMap[ImGuiKey_Tab] = 200;

	//for (int i = 0; i < IM_ARRAYSIZE(io.KeyMap); i++) {
	//	ImGui::Text("i: %d %d\n", i, io.KeyMap[i]);
	//}

	//ImGui::Text("Keys down:");      for (int i = 0; i < IM_ARRAYSIZE(io.KeysDown); i++) if (io.KeysDownDuration[i] >= 0.0f) { ImGui::SameLine(); ImGui::Text("%d (0x%X) (%.02f secs)", i, i, io.KeysDownDuration[i]); }
	//ImGui::Text("Keys pressed:");   for (int i = 0; i < IM_ARRAYSIZE(io.KeysDown); i++) if (ImGui::IsKeyPressed(i)) { ImGui::SameLine(); ImGui::Text("%d (0x%X)", i, i); }
	//ImGui::Text("Keys release:");   for (int i = 0; i < IM_ARRAYSIZE(io.KeysDown); i++) if (ImGui::IsKeyReleased(i)) { ImGui::SameLine(); ImGui::Text("%d (0x%X)", i, i); }
	//ImGui::Text("Keys mods: %s%s%s%s", io.KeyCtrl ? "CTRL " : "", io.KeyShift ? "SHIFT " : "", io.KeyAlt ? "ALT " : "", io.KeySuper ? "SUPER " : "");
	//ImGui::Text("Chars queue:");    for (int i = 0; i < io.InputQueueCharacters.Size; i++) { ImWchar c = io.InputQueueCharacters[i]; ImGui::SameLine();  ImGui::Text("\'%c\' (0x%04X)", (c > ' ' && c <= 255) ? (char)c : '?', c); } // FIXME: We should convert 'c' to UTF-8 here but the functions are not public.

	//ImGui::Text("NavInputs down:"); for (int i = 0; i < IM_ARRAYSIZE(io.NavInputs); i++) if (io.NavInputs[i] > 0.0f) { ImGui::SameLine(); ImGui::Text("[%d] %.2f", i, io.NavInputs[i]); }
	//ImGui::Text("NavInputs pressed:"); for (int i = 0; i < IM_ARRAYSIZE(io.NavInputs); i++) if (io.NavInputsDownDuration[i] == 0.0f) { ImGui::SameLine(); ImGui::Text("[%d]", i); }
	//ImGui::Text("NavInputs duration:"); for (int i = 0; i < IM_ARRAYSIZE(io.NavInputs); i++) if (io.NavInputsDownDuration[i] >= 0.0f) { ImGui::SameLine(); ImGui::Text("[%d] %.2f", i, io.NavInputsDownDuration[i]); }

}