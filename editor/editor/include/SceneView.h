#ifndef __SCENE_VIEW_H__
#define __SCENE_VIEW_H__

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

			void render(const char* textureNames[], const GLuint textures[], int count, PhysicsEngine::GraphicsQuery query, bool isOpenedThisFrame);

			bool isFocused() const;
	};
}

#endif
