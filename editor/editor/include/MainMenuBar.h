#ifndef __MAIN_MENU_BAR_H__
#define __MAIN_MENU_BAR_H__

namespace PhysicsEditor
{
	class MainMenuBar
	{
		private:
			bool newClicked;
			bool openClicked;
			bool saveClicked;
			bool saveAsClicked;

			bool newProjectClicked;
			bool openProjectClicked;
			bool saveProjectClicked;

			bool buildClicked;

			bool quitClicked;
			bool openInspectorClicked;
			bool openHierarchyClicked;
			bool openConsoleClicked;
			bool aboutClicked;

		public:
			MainMenuBar();
			~MainMenuBar();

			void render();

			bool isNewClicked() const;
			bool isOpenClicked() const;
			bool isSaveClicked() const;
			bool isSaveAsClicked() const;
			bool isBuildClicked() const;
			bool isQuitClicked() const;
			bool isNewProjectClicked() const;
			bool isOpenProjectClicked() const;
			bool isOpenInspectorCalled() const;
			bool isOpenHierarchyCalled() const;
			bool isOpenConsoleCalled() const;
			bool isAboutClicked() const;

		private:
			void showMenuFile();
			void showMenuEdit();
			void showMenuWindow();
			void showMenuHelp();
	};
}

#endif