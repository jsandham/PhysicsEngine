#ifndef __INSPECTOR_H__
#define __INSPECTOR_H__

#include <vector>

#include "InspectorDrawer.h"
#include "EditorScene.h"
#include "EditorClipboard.h"

#include "core/World.h"
#include "core/Entity.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
	class Inspector
	{
		public:
			Inspector();
			~Inspector();

			void render(World* world, EditorScene& scene, EditorClipboard& clipboard, bool isOpenedThisFrame);
			
		private:
			void drawEntity(World* world, EditorScene& scene, EditorClipboard& clipboard);
			void drawAsset(World* world, EditorScene& scene, EditorClipboard& clipboard);
			void drawCodeFile(World* world, EditorScene& scene, EditorClipboard& clipboard);

			// move to imgui extensions?
			bool BeginAddComponentDropdown(std::string filter, std::string& componentToAdd);
			void EndAddComponentDropdown();
	};
}

#endif