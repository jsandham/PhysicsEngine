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

#include "..//include/imgui_extensions.h"

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

	// enable docking
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	//io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

	//Init Win32
	ImGui_ImplWin32_Init(window);

	//Init OpenGL Imgui Implementation
	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 330";
	ImGui_ImplOpenGL3_Init(glsl_version);

	//Set Window bg color
	ImVec4 clear_color = ImVec4(1.000F, 1.000F, 1.000F, 1.0F);

	// Setup style
	ImGui::StyleColorsClassic();

	// add physics, render, and cleanup system to world
	world.addSystem<EditorCameraSystem>(0);
	renderSystem = world.addSystem<RenderSystem>(1);
	world.addSystem<CleanUpSystem>(2);

	renderSystem->renderToScreen = false;

	for (int i = 0; i < world.getNumberOfSystems(); i++) {
		System* system = world.getSystemByIndex(i);

		system->init(&world);
	}
	
	// add editor camera
	Entity* cameraEntity = world.createEntity();
	cameraEntity->doNotDestroy = true;

	Transform* transform = cameraEntity->addComponent<Transform>(&world);
	camera = cameraEntity->addComponent<Camera>(&world);

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

	ImGui::EnableDocking();

	ImGui::ShowDemoWindow();
	//ImGui::ShowMetricsWindow();
	//ImGui::Text(currentProjectPath.c_str());
	//ImGui::Text(currentScenePath.c_str());

	libraryDirectory.update(currentProjectPath);

	updateAssetsLoadedInWorld();

	std::string message = std::to_string(world.getNumberOfEntities());
	ImGui::Text(message.c_str());

	mainMenu.render(currentProjectPath);

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
	bool sceneViewOpenedThisFrame = mainMenu.isOpenSceneViewCalled();
	bool projectViewOpenedThisFrame = mainMenu.isOpenProjectViewCalled();
	
	hierarchy.render(world, hierarchyOpenedThisFrame);
	inspector.render(world, hierarchy.getSelectedEntity(), inspectorOpenedThisFrame);
	console.render(consoleOpenedThisFrame);
	projectView.render(currentProjectPath, projectViewOpenedThisFrame);

	updateInputPassedToSystems(&input);

	for (int i = 0; i < world.getNumberOfSystems(); i++) {
		System* system = world.getSystemByIndex(i);

		system->update(input);
	}

	GLuint colorTex = renderSystem->getColorTexture();
	GLuint depthTex = renderSystem->getDepthTexture();
	GLuint normalTex = renderSystem->getNormalTexture();

	const char* textureNames[] = { "Color", "Depth", "Normals" };
	const GLuint textures[] = {colorTex, depthTex, normalTex};
	
	GraphicsQuery query = renderSystem->getGraphicsQuery();

	sceneView.render(textureNames, textures, 3, query, sceneViewOpenedThisFrame);
	
	aboutPopup.render(mainMenu.isAboutClicked());

	if (mainMenu.isQuitClicked()) {
		quitCalled = true;
	}
	
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
	world.latentDestroyEntitiesInWorld(); // need to destroy assets too!

	// re-centre editor camera to default position
	camera->position = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->front = glm::vec3(1.0f, 0.0f, 0.0f);
	camera->up = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->backgroundColor = glm::vec4(0.15, 0.15f, 0.15f, 1.0f);
	camera->updateInternalCameraState();
}

void Editor::openScene(std::string path)
{
	// check to make sure the scene is part of the current project
	if (path.find(currentProjectPath + "\\data\\") != 0) {
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

	// mark any (non-editor) entities in currently opened scene to be latent destroyed
	world.latentDestroyEntitiesInWorld(); // need to destroy assets too!

	// reset editor camera
	camera->position = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->front = glm::vec3(1.0f, 0.0f, 0.0f);
	camera->up = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->backgroundColor = glm::vec4(0.15, 0.15f, 0.15f, 1.0f);
	camera->updateInternalCameraState();

	// load binary version of scene into world (ignoring systems and cameras)
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

	// mark any (non-editor) entities in currently opened scene to be latent destroyed
	world.latentDestroyEntitiesInWorld(); // need to destroy assets too!

	// reset editor camera
	camera->position = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->front = glm::vec3(1.0f, 0.0f, 0.0f);
	camera->up = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->backgroundColor = glm::vec4(0.15, 0.15f, 0.15f, 1.0f);
	camera->updateInternalCameraState();
}

void Editor::openProject(std::string path)
{
	currentProjectPath = path;

	assetsAddedToWorld.clear();

	// mark any (non-editor) entities in currently opened scene to be latent destroyed
	world.latentDestroyEntitiesInWorld(); 

	// reset editor camera
	camera->position = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->front = glm::vec3(1.0f, 0.0f, 0.0f);
	camera->up = glm::vec3(0.0f, 0.0f, 1.0f);
	camera->backgroundColor = glm::vec4(0.15, 0.15f, 0.15f, 1.0f);
	camera->updateInternalCameraState();
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

// TODO: This is platform specific. Should move into main?
void Editor::updateInputPassedToSystems(Input* input)
{
	ImGuiIO& io = ImGui::GetIO();

	for (int i = 0; i < 61; i++) {
		input->keyWasDown[i] = input->keyIsDown[i];
		input->keyIsDown[i] = false;
	}

	for (int i = 0; i < 5; i++) {
		input->mouseButtonWasDown[i] = input->mouseButtonIsDown[i];
		input->mouseButtonIsDown[i] = false;
	}

	for (int i = 0; i < 14; i++) {
		input->xboxButtonWasDown[i] = input->xboxButtonIsDown[i];
		input->xboxButtonIsDown[i] = false;
	}

	if (sceneView.isFocused()) {
			// 0 - 9
		for (int i = 0; i < 10; i++) {
			input->keyIsDown[0] = io.KeysDown[48 + i];
		}

		// A - Z
		for (int i = 0; i < 26; i++) {
			input->keyIsDown[10 + i] = io.KeysDown[65 + i];
		}

		input->keyIsDown[36] = io.KeysDown[13]; // Enter
		input->keyIsDown[37] = io.KeysDown[38]; // Up
		input->keyIsDown[38] = io.KeysDown[40]; // Down
		input->keyIsDown[39] = io.KeysDown[37]; // Left
		input->keyIsDown[40] = io.KeysDown[39]; // Right
		input->keyIsDown[41] = io.KeysDown[32]; // Space
		input->keyIsDown[42] = io.KeysDown[16]; // LShift
		input->keyIsDown[43] = io.KeysDown[16]; // RShift
		input->keyIsDown[44] = io.KeysDown[9];  // Tab
		input->keyIsDown[45] = io.KeysDown[8];  // Backspace
		input->keyIsDown[46] = io.KeysDown[20]; // CapsLock
		input->keyIsDown[47] = io.KeysDown[17]; // LCtrl
		input->keyIsDown[48] = io.KeysDown[17]; // RCtrl
		input->keyIsDown[49] = io.KeysDown[27]; // Escape
		input->keyIsDown[50] = io.KeysDown[45]; // NumPad0
		input->keyIsDown[51] = io.KeysDown[35]; // NumPad1
		input->keyIsDown[52] = io.KeysDown[40]; // NumPad2
		input->keyIsDown[53] = io.KeysDown[34]; // NumPad3
		input->keyIsDown[54] = io.KeysDown[37]; // NumPad4
		input->keyIsDown[55] = io.KeysDown[12]; // NumPad5
		input->keyIsDown[56] = io.KeysDown[39]; // NumPad6
		input->keyIsDown[57] = io.KeysDown[36]; // NumPad7
		input->keyIsDown[58] = io.KeysDown[8];  // NumPad8
		input->keyIsDown[59] = io.KeysDown[33]; // NumPad9
	}

	if(sceneView.isFocused())
	{
		input->mouseButtonIsDown[0] = io.MouseDown[0]; // Left Mouse Button
		input->mouseButtonIsDown[1] = io.MouseDown[2]; // Middle Mouse Button
		input->mouseButtonIsDown[2] = io.MouseDown[1]; // Right Mouse Button
		input->mouseButtonIsDown[3] = io.MouseDown[3]; // Alt0 Mouse Button
		input->mouseButtonIsDown[4] = io.MouseDown[4]; // Alt1 Mouse Button

		input->mouseDelta = io.MouseWheel;
		input->mousePosX = (int)io.MousePos.x;
		input->mousePosY = (int)io.MousePos.y;
	}
}