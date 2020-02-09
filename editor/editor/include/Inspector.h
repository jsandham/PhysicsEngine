#ifndef __INSPECTOR_H__
#define __INSPECTOR_H__

#include <vector>

#include "InspectorDrawer.h"
#include "EditorProject.h"
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

			void render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, bool isOpenedThisFrame);
			
		private:
			void drawEntity(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard);

			template<class T>
			void drawAsset(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard)
			{
				T* asset = world->getAsset<T>(clipboard.getSelectedId());

				InspectorDrawer* drawer = loadInternalInspectorAssetDrawer(AssetType<T>::type);

				drawer->render(world, project, scene, clipboard, asset->assetId);
				ImGui::Separator();

				delete drawer;
			}









			void drawCodeFile(World* world, EditorScene& scene, EditorClipboard& clipboard);

			// move to imgui extensions?
			bool BeginAddComponentDropdown(std::string filter, std::string& componentToAdd);
			void EndAddComponentDropdown();
	};
}

#endif