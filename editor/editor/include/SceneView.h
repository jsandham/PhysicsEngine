#ifndef __SCENE_VIEW_H__
#define __SCENE_VIEW_H__

#include <queue>

#include "PerformanceQueue.h"

#include "core/World.h"

#include "systems/RenderSystem.h"
#include "graphics/GraphicsQuery.h"
//#include "graphics/GraphicsTargets.h"

#include "EditorCameraSystem.h"
#include "EditorClipboard.h"

#include "imgui.h"

namespace PhysicsEditor
{
	class SceneView
	{
		private:
			bool focused;
			bool hovered;
			PerformanceQueue perfQueue;

			ImVec2 windowPos;
			ImVec2 sceneContentMin;
			ImVec2 sceneContentMax;

		public:
			SceneView();
			~SceneView();

			void render(PhysicsEngine::World* world, 
						PhysicsEngine::EditorCameraSystem* cameraSystem, 
						EditorClipboard& clipboard, 
						bool isOpenedThisFrame);

			bool isFocused() const;
			bool isHovered() const;

			ImVec2 getSceneContentMin() const;
			ImVec2 getSceneContentMax() const;
			ImVec2 getWindowPos() const;

	};
}

#endif
