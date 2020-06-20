#ifndef __DEFERREDRENDERER_H__
#define __DEFERREDRENDERER_H__

#include <vector>
#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/World.h"
#include "../core/Input.h"

#include "DeferredRendererState.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
	class DeferredRenderer
	{
		private:
			World* mWorld;

			DeferredRendererState mState;

		public:
			DeferredRenderer();
			~DeferredRenderer();

			void init(World* world, bool renderToScreen);
			void update(Input input, Camera* camera, std::vector<RenderObject>& renderObjects);
	};
}

#endif