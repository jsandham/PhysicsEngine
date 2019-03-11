#ifndef __RENDERER_H__
#define __RENDERER_H__

#include "../core/World.h"

#include "GLState.h"
#include "BatchManager.h"

namespace PhysicsEngine
{
	class Renderer
	{
		private:
			World* world;

			// internal graphics state
			GLCamera cameraState;
			GLDirectionalLight directionLightState;
			GLSpotLight spotLightState;
			GLPointLight pointLightState;

			BatchRenderer batchManager;

			unsigned int pass;

		public:
			Renderer();
			~Renderer();

			void init(World* world);
			void update();
	};
}

#endif