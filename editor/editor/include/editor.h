#ifndef __EDITOR_H__
#define __EDITOR_H__

#include <windows.h>
#include <string>

#include "core/World.h"

#include "AssetDirectory.h"

#include "MainMenuBar.h"
#include "Inspector.h"
#include "Hierarchy.h"
#include "Project.h"
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

			AssetDirectory assetDirectory;

			MainMenuBar mainMenu;
			Inspector inspector;
			Hierarchy hierarchy;
			Project project;

			Filebrowser filebrowser;
			ProjectWindow projectWindow; //ProjectBrowser?
			BuildWindow buildWindow;
			AboutPopup aboutPopup;

			bool quitCalled;
			bool isInspectorVisible;
			bool isHierarchyVisible;

			std::string currentProjectPath;
			std::string currentScenePath;

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
			void newScene(std::string path);
			void openScene(std::string path);
			void createProject(std::string path);
			void openProject(std::string path);
	};
}

#endif