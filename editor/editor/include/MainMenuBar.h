#ifndef __MAIN_MENU_BAR_H__
#define __MAIN_MENU_BAR_H__

#include "Filebrowser.h"

namespace PhysicsEditor
{
	class MainMenuBar
	{
		private:
			Filebrowser filebrowser;

		public:
			MainMenuBar();
			~MainMenuBar();

			void render();

		private:
			void ShowMenuFile();
			void ShowMenuEdit();
	};
}

#endif