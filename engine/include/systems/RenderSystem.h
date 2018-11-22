#ifndef __RENDERSYSTEM_H__
#define __RENDERSYSTEM_H__

#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "System.h"

#include "../graphics/GLState.h"
#include "../graphics/GLHandle.h"

#include "../core/Material.h"
#include "../core/Shader.h"
#include "../core/Texture2D.h"
#include "../core/PerformanceGraph.h"
#include "../core/DebugWindow.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/type_ptr.hpp"
#include "../glm/gtc/matrix_transform.hpp"

namespace PhysicsEngine
{
	class RenderSystem : public System
	{
		private:
			unsigned int pass;

			// internal graphics state
			GLCamera cameraState;
			GLDirectionalLight directionLightState;
			GLSpotLight spotLightState;
			GLPointLight pointLightState;

			PerformanceGraph* graph;
			DebugWindow* window;

			Material* graphMaterial;
			Material* windowMaterial;
			Shader* graphShader;
			Shader* normalMapShader;
			Texture2D* normalMap;

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