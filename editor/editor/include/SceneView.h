#ifndef __SCENE_VIEW_H__
#define __SCENE_VIEW_H__

#include <queue>

#include "PerformanceQueue.h"

#include "core/World.h"

#include "systems/RenderSystem.h"
#include "graphics/GraphicsQuery.h"
#include "graphics/GraphicsTargets.h"

#include "EditorCameraSystem.h"

namespace PhysicsEditor
{
	class SceneView
	{
		private:
			bool focused;
			bool hovered;
			PerformanceQueue perfQueue;

		public:
			SceneView();
			~SceneView();

			void render(PhysicsEngine::World* world, PhysicsEngine::EditorCameraSystem* cameraSystem, PhysicsEngine::GraphicsTargets targets, PhysicsEngine::GraphicsQuery query, bool isOpenedThisFrame);

			bool isFocused() const;
			bool isHovered() const;
	};
}

#endif
