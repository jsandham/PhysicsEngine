#ifndef __EDITOR_H__
#define __EDITOR_H__

#include <windows.h>

#include "core/World.h"

#include "MainMenuBar.h"
#include "Inspector.h"
#include "Hierarchy.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
	class Editor
	{
		private:
			World world;

			MainMenuBar mainMenu;
			Inspector inspector;
			Hierarchy hierarchy;

			bool quitCalled;
			bool isInspectorVisible;
			bool isHierarchyVisible;

		public:
			Editor();
			~Editor();

			void init(HWND window, int width, int height);
			void cleanUp();
			void render();

			bool isQuitCalled();
	};
}

#endif