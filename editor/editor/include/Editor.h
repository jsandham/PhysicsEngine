#ifndef __EDITOR_H__
#define __EDITOR_H__

#include <windows.h>
#include <string>
#include <unordered_set>

#include "core/World.h"
#include "components/Camera.h"

#include "LibraryDirectory.h"

#include "EditorProject.h"
#include "EditorScene.h"
#include "EditorMenuBar.h"
#include "EditorToolbar.h"
#include "Inspector.h"
#include "Hierarchy.h"
#include "ProjectView.h"
#include "Console.h"
#include "SceneView.h"
#include "Filebrowser.h"
#include "ProjectWindow.h"
#include "BuildWindow.h"
#include "PreferencesWindow.h"
#include "AboutPopup.h"
#include "CommandManager.h"

#include "systems/RenderSystem.h"

namespace PhysicsEditor
{
	class Editor
	{
		private:
			World world;

			LibraryDirectory libraryDirectory;

			EditorMenuBar editorMenu;
			EditorToolbar editorToolbar;
			Inspector inspector;
			Hierarchy hierarchy;
			ProjectView projectView;
			Console console;
			SceneView sceneView;

			Filebrowser filebrowser;
			ProjectWindow projectWindow; //ProjectBrowser? ProjectPopup?
			BuildWindow buildWindow;
			PreferencesWindow preferencesWindow;
			AboutPopup aboutPopup;

			CommandManager commandManager;

			EditorProject currentProject;
			EditorScene currentScene;
			
			Input input;
			Camera* camera;
			RenderSystem* renderSystem;

			std::unordered_set<std::string> assetsAddedToWorld;

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
			void updateProjectAndSceneState();
			void updateAssetsLoadedInWorld();
			void updateInputPassedToSystems(Input* input);
	};
}

#endif