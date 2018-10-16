#ifndef __RENDERSYSTEM_H__
#define __RENDERSYSTEM_H__

#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "System.h"

#include "../graphics/GLState.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/type_ptr.hpp"
#include "../glm/gtc/matrix_transform.hpp"

namespace PhysicsEngine
{
	class RenderSystem : public System
	{
		private:
			unsigned int numLights;
			unsigned int pass;

			// camera data   //  Probably can get rid of this now that it is contained inside GLCamera???
			glm::mat4 view;
			glm::mat4 projection;
			glm::vec3 cameraPos;

			// internal graphics state
			GLCamera cameraState;
			GLDirectionalLight directionLightState;
			GLSpotLight spotLightState;
			GLPointLight pointLightState;

		public:
			RenderSystem();
			RenderSystem(unsigned char* data);
			~RenderSystem();

			void init();
			void update();

		private:
			void renderScene();
	};
}

#endif