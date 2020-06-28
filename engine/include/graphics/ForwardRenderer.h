#ifndef __FORWARDRENDERER_H__
#define __FORWARDRENDERER_H__

#include <vector>
#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/World.h"
#include "../core/Input.h"

#include "ForwardRendererState.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
	class ForwardRenderer
	{
		private:
			World* mWorld;

			ForwardRendererState mState;

		public:
			ForwardRenderer();
			~ForwardRenderer();

			void init(World* world, bool renderToScreen);
			void update(Input input, 
						Camera* camera, 
						const std::vector<std::pair<uint64_t, int>>& renderQueue, 
						const std::vector<RenderObject>& renderObjects);
	};
}

#endif