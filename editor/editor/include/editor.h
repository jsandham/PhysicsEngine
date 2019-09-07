#ifndef __EDITOR_H__
#define __EDITOR_H__

#include <windows.h>
#include <string>
#include <unordered_set>

#include "core/World.h"

#include "LibraryDirectory.h"

#include "MainMenuBar.h"
#include "Inspector.h"
#include "Hierarchy.h"
#include "Project.h"
#include "Console.h"
#include "Filebrowser.h"
#include "ProjectWindow.h"
#include "BuildWindow.h"
#include "AboutPopup.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
	class Editor
	{
		private:
			World world;

			LibraryDirectory libraryDirectory;

			MainMenuBar mainMenu;
			Inspector inspector;
			Hierarchy hierarchy;
			Project project;
			Console console;

			Filebrowser filebrowser;
			ProjectWindow projectWindow; //ProjectBrowser? ProjectPopup?
			BuildWindow buildWindow;
			AboutPopup aboutPopup;

			bool quitCalled;
			
			std::string currentProjectPath;
			std::string currentScenePath;

			Camera* camera;
			std::vector<Guid> editorEntityIds;

			std::unordered_set<std::string> assetsAddedToWorld;

		public:
			Editor();
			~Editor();

			void init(HWND window, int width, int height);
			void cleanUp();
			void render();

			bool isQuitCalled() const;
			std::string getCurrentProjectPath() const;
			std::string getCurrentScenePath() const;

		private:
			void newScene();
			void openScene(std::string path);
			void saveScene(std::string path);
			void createProject(std::string path);
			void openProject(std::string path);
			void updateAssetsLoadedInWorld();
	};
}

#endif