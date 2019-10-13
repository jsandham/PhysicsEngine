#ifndef __SCENE_VIEW_H__
#define __SCENE_VIEW_H__

#include "core/World.h"

#include "systems/RenderSystem.h"
#include "graphics/GraphicsQuery.h"

namespace PhysicsEditor
{
	class SceneView
	{
		private:
			bool focused;

		public:
			SceneView();
			~SceneView();

			void render(PhysicsEngine::World* world, const char* textureNames[], const GLint textures[], int count, PhysicsEngine::GraphicsQuery query, bool isOpenedThisFrame);

			bool isFocused() const;
	};
}

#endif
