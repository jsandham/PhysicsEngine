#ifndef __SCENE_VIEW_H__
#define __SCENE_VIEW_H__

#include "systems/RenderSystem.h"

namespace PhysicsEditor
{
	class SceneView
	{
		public:
			SceneView();
			~SceneView();

			void render(GLuint mainTexture, bool isOpenedThisFrame);
	};
}

#endif
