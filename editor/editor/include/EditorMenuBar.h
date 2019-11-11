#ifndef __MAIN_MENU_BAR_H__
#define __MAIN_MENU_BAR_H__

#include <string>

#include "EditorProject.h"
#include "EditorScene.h"

namespace PhysicsEditor
{
	class EditorMenuBar
	{
		private:
			bool projectSelected;
			bool sceneSelected;

			// File
			bool newSceneClicked;
			bool openSceneClicked;
			bool saveClicked;
			bool saveAsClicked;
			bool newProjectClicked;
			bool openProjectClicked;
			bool saveProjectClicked;
			bool buildClicked;
			bool quitClicked;

			// Edit
			bool preferencesClicked;

			// Windows
			bool openInspectorClicked;
			bool openHierarchyClicked;
			bool openConsoleClicked;
			bool openSceneViewClicked;
			bool openProjectViewClicked;

			// About
			bool aboutClicked;

		public:
			EditorMenuBar();
			~EditorMenuBar();

			void render(const EditorProject project, const EditorScene scene);

			bool isNewSceneClicked() const;
			bool isOpenSceneClicked() const;
			bool isSaveClicked() const;
			bool isSaveAsClicked() const;
			bool isBuildClicked() const;
			bool isQuitClicked() const;
			bool isNewProjectClicked() const;
			bool isOpenProjectClicked() const;
			bool isOpenInspectorCalled() const;
			bool isOpenHierarchyCalled() const;
			bool isOpenConsoleCalled() const;
			bool isOpenSceneViewCalled() const;
			bool isOpenProjectViewCalled() const;
			bool isAboutClicked() const;
			bool isPreferencesClicked() const;

		private:
			void showMenuFile();
			void showMenuEdit();
			void showMenuWindow();
			void showMenuHelp();
	};
}

#endif