#ifndef __MAIN_MENU_BAR_H__
#define __MAIN_MENU_BAR_H__

#include "Filebrowser.h"
#include "AboutPopup.h"

namespace PhysicsEditor
{
	class MainMenuBar
	{
		private:
			Filebrowser filebrowser;
			AboutPopup aboutPopup;

			bool newClicked;
			bool openClicked;
			bool saveClicked;
			bool saveAsClicked;
			bool quitClicked;
			bool openInspectorClicked;
			bool openHierarchyClicked;
			bool aboutClicked;

		public:
			MainMenuBar();
			~MainMenuBar();

			void render();

			bool isNewClicked();
			bool isOpenClicked();
			bool isSaveClicked();
			bool isSaveAsClicked();
			bool isQuitClicked();
			bool isOpenInspectorCalled();
			bool isOpenHierarchyCalled();
			bool isAboutClicked();

		private:
			void showMenuFile();
			void showMenuEdit();
			void showMenuWindow();
			void showMenuHelp();
	};
}

#endif